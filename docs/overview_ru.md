# RL Episode Inspector — как устроен инструмент (кратко)

Документ для быстрого понимания. Только важное. Подробности — в коде по ссылкам.

## 1. Что это

Инструмент для **покадрового разбора и 3D-реплея эпизодов RL** из NVIDIA Isaac Lab.
Записываем эпизод (сигналы + награды + позы всех тел робота) на диск, потом
смотрим в браузере: таймлайн, графики наград, 3D-сцена с настоящей геометрией
робота. Просмотр работает **без Isaac** — нужны только записанные файлы.

Ключевая идея: ядро **ничего не знает про конкретную задачу**. Оно оперирует
обобщёнными «сигналами» (`SignalSpec`). Cartpole, Franka, Humanoid — это
адаптеры в `examples/`, которые просто скармливают значения рекордеру.

## 2. Архитектура и поток данных

```
Isaac Lab env ──▶ адаптер (examples/) ──▶ EpisodeRecorder ──▶ диск
                                                              (metadata.json + frames.parquet + assets/*.glb)
                                                                    │
браузер (React/Three.js) ◀── FastAPI (server/) ◀───────────────────┘
```

- **Запись** (Python): адаптер на каждом шаге зовёт `record_frame(...)`, в конце
  `end_episode()` пишет валидированный эпизод.
- **Хранение**: одна папка на эпизод. Меши робота (`.glb`) — отдельно в `assets/`.
- **Чтение**: FastAPI отдаёт метаданные/кадры/сигналы как JSON, меши — как файлы.
- **UI**: SPA на React + Zustand; 3D — Three.js.

## 3. Модель данных (самое важное)

### Сигнал (`storage/signal_schema.py`)
`SignalSpec`: `name`, `kind`, `unit`, `description`. `kind` (`SignalKind`):
`state`, `action`, `observation`, `reward_raw`, `reward_weighted`,
`reward_total`, `pose`, `debug`, `event`. По `kind` UI понимает, что это и куда
рисовать. Набор сигналов **выводится из ключей**, переданных в `record_frame` —
руками схему писать не надо.

### Кадры на диске — `frames.parquet`
Одна строка = один кадр, колонки = сигналы. Обязательные колонки
(`storage/schemas.py`): `frame_index`, `timestamp`, `terminated`, `truncated`,
`done`.

### Декомпозиция награды (`recorder/reward_buffer.py`) — фиксированные имена колонок:
```
reward_<name>_raw        # сырой член до веса
reward_<name>_weighted   # raw * weight
reward_step_total        # сумма всех weighted за кадр
reward_cumulative        # бегущая сумма step_total
```
Это позволяет в UI показывать вклад каждого члена награды по отдельности.

### Поза тела (для 3D) — 7 колонок на тело:
```
pose_<body>_{px,py,pz,qw,qx,qy,qz}   # позиция + кватернион (порядок wxyz)
```
См. `pose_columns()` в `recorder/episode_recorder.py`.

### `metadata.json` (`EpisodeMetadata`)
Содержит сводку (`episode_return`, `num_frames`, `dt/fps`, флаги завершения),
список `signals` и `viewer` (`ViewerSpec`) — описание, **как рисовать 3D**:
- `type`: какой вьювер (`articulation3d`, `reach`, и т.п.);
- `bodies`: список `BodySpec` (`name`, `parent`, имена колонок `pos`/`quat`,
  путь к мешу `mesh`);
- `markers`: точечные маркеры (например, цель reach);
- `up_axis`: какая ось «вверх» в записанных позах (`"z"` для Isaac).

## 4. Запись эпизодов — `EpisodeRecorder`

`recorder/episode_recorder.py`. Жизненный цикл:
1. `__init__(task_name, dt, viewer_type, up_axis, ...)`.
2. (для 3D) `register_bodies(names, parents, meshes=[...])` — описывает скелет.
3. `start_episode(...)`.
4. на каждом шаге: `record_frame(frame_index, timestamp, state=..., action=...,
   rewards_raw=..., reward_weights=..., poses={body:(px,py,pz,qw,qx,qy,qz)},
   terminated, truncated)`.
5. `end_episode(reset_reason)` → валидирует и пишет на диск.

Всё нефинитное (NaN/inf) отбрасывается с ошибкой (`_finite`). Адаптер задачи
отвечает за извлечение значений из своего окружения.

## 5. Backend — FastAPI (`server/`)

Эндпоинты (`server/routes.py`):
- `GET /api/episodes` — список (сводки).
- `GET /api/episodes/{id}/metadata` — метаданные.
- `GET /api/episodes/{id}/frames?start&end&names` — колонки кадров (срез/выборка
  колонок).
- `GET /api/episodes/{id}/signals?names` — описания сигналов + ряды значений.
- `GET /api/ranking?mode=best|worst|median` — выбор эпизода по `episode_return`
  (`ranking/episode_ranker.py`).
- `GET /api/assets/<path>` — отдача `.glb`. Защита от path-traversal
  (`server/security.py`, `safe_subpath`).

## 6. Frontend (`frontend/src/`)

### Состояние плеера — `playback/playbackStore.ts` (Zustand)
Единый источник правды. Главное:
- `loaded` — `{metadata, columns}` текущего эпизода;
- `currentFrame` (может быть дробным при проигрывании), `isPlaying`, `loop`,
  `speed`;
- `renderMode: "models" | "cubes"` — настоящие меши или прокси-кубы;
- экшены: `selectEpisode`, `play/pause/togglePlay`, `seek`, `stepFrames`,
  `tick(dt)` и т.д.

**Ни один компонент не хранит номер кадра у себя** — все читают `currentFrame`
из стора. Графики, панели значений и 3D синхронизированы автоматически.

### Графики — свои SVG-компоненты (`TimeSeriesChart`, `RewardCharts`, ...),
клик/драг по графику = seek по времени.

### 3D-вьювер — `components/ArticulationViewer.tsx`
Three.js. На каждое тело — своя `THREE.Group`, в неё грузится `.glb` (GLTFLoader)
или прокси. Каждый кадр:
```
g.position.set(px, py, pz);
g.quaternion.set(qx, qy, qz, qw);   // записано wxyz → Three хочет xyzw
```
Важные детали производительности (под софтверным WebGL это критично):
- `antialias:false`, `pixelRatio ≤ 1`;
- **рендер по требованию**: rAF-цикл останавливается, когда ничего не меняется
  (пауза/камера успокоилась) — нет 60 fps вхолостую;
- авто-кадрирование по bbox **нулевого кадра** + follow-камера за корневым телом;
- ось z-вверх (Isaac) → y-вверх (Three) делается поворотом корневой группы
  `rotation.x = -π/2`, без поворота каждого тела вручную.

## 7. 3D-реплей: системы координат — самое тонкое место

Цель: робот в визуализации выглядит **точно как в Isaac** (настоящая геометрия,
без ручной настройки). Меши экспортируются автоматически из USD.

### Экспорт мешей — `examples/export_env_meshes.py` + `examples/scene_geometry.py`
Бутстрапит задачу, обходит USD-стейдж, тесселирует **и Mesh, и примитивы**
(Capsule/Sphere/Cylinder/Cube/Cone), пишет по одному `.glb` на тело. Геометрия
запекается **в локальную систему тела** (link-frame), чтобы во вьювере её можно
было ставить по записанной позе.

Флаг `--frame`:
- `physics` — печь относительно живой позы физики (`body_pos_w/body_quat_w`).
  Корректно, когда поза при `reset()` совпадает с авторской позой USD (Franka).
- `usd` — печь относительно **авторского трансформа link-прима** (не зависит от
  позы). Нужно, когда `reset()` ставит робота в другую позу, чем в USD
  (Humanoid AMP сбрасывается в кадр mocap).

### Ключевые грабли (которые чинили) — почему важно
1. **Единицы стейджа.** `body_pos_w` всегда в метрах, а USD-трансформы — в
   единицах стейджа. При не-метровом стейдже трансляции не сокращаются. Решение:
   делить позиции тела на `meters_per_unit` при сборке матрицы (для метрового
   стейджа — no-op).
2. **Кватернионы warp vs torch.** Newton-бэкенд хранит кватернионы как `xyzw`,
   а Isaac документирует `wxyz`. Везде приводим warp-кватернионы перестановкой
   `[3,0,1,2]` (`examples/isaaclab_poses.py`). Рекордер хранит `wxyz`.
3. **Коллизия имён link↔geom.** Видимый geom-прим часто называется так же, как
   его link (`.../right_upper_arm/right_upper_arm`). При запекании нельзя брать
   за «систему тела» прим с тем же именем-потомок — иначе сократится собственный
   ~90° поворот geom'а (например, тот, что делает капсулу голени вертикальной), и
   геометрия запечётся в системе geom'а, а не link'а — все части окажутся
   повёрнуты на ~90°. Решение в `scene_geometry.py`: брать **самый верхний**
   предок с именем тела (настоящий rigid-body link).

После этих фиксов кватернионы PhysX совпадают с системой запечённого меша →
во вьювере **прямое применение кватернiona** даёт точную позу (никаких
костылей-«ориентаций по кости» не нужно). Проверено через диагностику
`examples/inspect_humanoid_frames.py` (сравнение физической и авторской систем в
дефолтной позе: они совпадают, а 90° сидит в трансформе самого geom'а).

## 8. Как добавить новую задачу (шпаргалка)

1. Адаптер в `examples/<task>/`: достаёт из env'а state/action/награды/позы.
2. (для 3D) один раз: `export_env_meshes --task ... --frame usd|physics
   --out-dir sample_data/<task>/assets --robot-key <key>`.
3. Создать `EpisodeRecorder(viewer_type="articulation3d", up_axis="z")`,
   `register_bodies(names, parents, meshes=[f"<key>/{n}.glb" ...])`, гонять
   эпизод через `record_frame(..., poses=...)`, `end_episode()`.
4. Сложить эпизоды в `sample_data/<task>/episodes`, запустить
   `rl-episode-inspector serve --episodes-dir ...`.

## 9. Где что лежит

- Ядро записи/хранения: `python/rl_episode_inspector/{recorder,storage}/`.
- Сервер: `python/rl_episode_inspector/server/`.
- Адаптеры задач и экспорт геометрии: `python/rl_episode_inspector/examples/`.
- UI: `frontend/src/` (стор — `playback/`, 3D — `components/ArticulationViewer.tsx`).
- Расширенная архитектура (EN): `docs/architecture.md`.
