// Lightweight interactive time-series chart (SVG).
//
// Why custom instead of Plotly/ECharts: the headline requirement is precise
// click-to-seek and drag-to-scrub with a marker that follows the cursor. Mapping
// pointer-x -> frame directly (via frameSync) is simpler and more reliable than
// adapting a charting lib's interaction model, and keeps the bundle tiny. The
// pixel<->frame math lives in frameSync.ts and is unit-tested.

import { useCallback, useEffect, useRef, useState } from "react";
import { frameToX, seriesExtent, xToFrame } from "../playback/frameSync";

export interface Series {
  name: string;
  color: string;
  values: number[];
}

interface Props {
  series: Series[];
  numFrames: number;
  currentFrame: number;
  onSeek: (frame: number) => void;
  height?: number;
  title?: string;
  /** Show per-series value of the current frame in a compact legend. */
  showLegend?: boolean;
}

const PAD = { left: 48, right: 12, top: 10, bottom: 22 };

function useMeasuredWidth(): [React.RefObject<HTMLDivElement>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setWidth(w);
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

export function TimeSeriesChart({
  series,
  numFrames,
  currentFrame,
  onSeek,
  height = 140,
  title,
  showLegend = true,
}: Props) {
  const [ref, width] = useMeasuredWidth();
  const dragging = useRef(false);

  const all = series.flatMap((s) => s.values);
  const { min, max } = seriesExtent(all);

  const yOf = (v: number) => {
    const t = (v - min) / (max - min || 1);
    return PAD.top + (1 - t) * (height - PAD.top - PAD.bottom);
  };
  const xOf = (frame: number) => frameToX(frame, numFrames, width, PAD.left, PAD.right);

  const seekFromEvent = useCallback(
    (clientX: number, el: SVGSVGElement) => {
      const rect = el.getBoundingClientRect();
      onSeek(xToFrame(clientX - rect.left, numFrames, width, PAD.left, PAD.right));
    },
    [numFrames, width, onSeek],
  );

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    dragging.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    seekFromEvent(e.clientX, e.currentTarget);
  };
  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragging.current) seekFromEvent(e.clientX, e.currentTarget);
  };
  const stopDrag = (e: React.PointerEvent<SVGSVGElement>) => {
    dragging.current = false;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* pointer already released */
    }
  };

  const frame = Math.round(currentFrame);
  const markerX = xOf(frame);

  const pathFor = (values: number[]) => {
    if (values.length === 0 || width <= 0) return "";
    let d = "";
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      d += `${d ? "L" : "M"}${xOf(i).toFixed(1)} ${yOf(v).toFixed(1)} `;
    }
    return d;
  };

  return (
    <div className="chart" ref={ref}>
      {title && <div className="chart-title">{title}</div>}
      <svg
        width={width}
        height={height}
        className="chart-svg"
        role="img"
        aria-label={title ?? "time series chart"}
        data-testid="timeseries-chart"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={stopDrag}
        onPointerLeave={stopDrag}
      >
        {/* y axis labels */}
        <text x={4} y={PAD.top + 8} className="axis-label">
          {max.toFixed(2)}
        </text>
        <text x={4} y={height - PAD.bottom} className="axis-label">
          {min.toFixed(2)}
        </text>
        {/* zero line if range straddles zero */}
        {min < 0 && max > 0 && (
          <line
            x1={PAD.left}
            x2={width - PAD.right}
            y1={yOf(0)}
            y2={yOf(0)}
            className="zero-line"
          />
        )}
        {/* series */}
        {series.map((s) => (
          <path key={s.name} d={pathFor(s.values)} fill="none" stroke={s.color} strokeWidth={1.5} />
        ))}
        {/* current-frame marker */}
        <line
          x1={markerX}
          x2={markerX}
          y1={PAD.top}
          y2={height - PAD.bottom}
          className="frame-marker"
        />
      </svg>
      {showLegend && (
        <div className="chart-legend">
          {series.map((s) => (
            <span key={s.name} className="legend-item">
              <span className="swatch" style={{ background: s.color }} />
              {s.name}
              <span className="legend-value">
                {Number.isFinite(s.values[frame]) ? s.values[frame]?.toFixed(3) : "—"}
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
