import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { useWebcam } from "./hooks/useWebcam.ts";
import { detectCapabilities } from "./gl/glUtils.ts";
import { Pipeline, type PipelineParams } from "./gl/Pipeline.ts";
import { FilterMask, type BrushMode } from "./filters/FilterMask.ts";

// Visible canvases are a fixed size; the FFT compute resolution is separate.
const DISPLAY = 512;
const FFT_SIZES = [256, 512, 1024] as const;

export default function App() {
  const { videoRef, status, error } = useWebcam();
  const grayRef = useRef<HTMLCanvasElement | null>(null);
  const spectrumRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const reconRef = useRef<HTMLCanvasElement | null>(null);
  const roundtripRef = useRef<HTMLCanvasElement | null>(null);
  const cursorRef = useRef<HTMLDivElement | null>(null);
  const caps = useMemo(() => detectCapabilities(), []);

  const [renderError, setRenderError] = useState<string | null>(null);
  const [logGain, setLogGain] = useState(0.08);
  const [colormap, setColormap] = useState(1);
  const [filterEnabled, setFilterEnabled] = useState(true);
  const [fftSize, setFftSize] = useState(512);
  const [fps, setFps] = useState(0);
  // Hiding a pane drops its GPU work: the pipeline skips any view whose canvas
  // context is null (the inverse FFT and re-FFT are the expensive passes).
  const [showRecon, setShowRecon] = useState(true);
  const [showRoundtrip, setShowRoundtrip] = useState(true);

  const [brushMode, setBrushMode] = useState<BrushMode>("block");
  const [brushSize, setBrushSize] = useState(28);
  const [cutoff, setCutoff] = useState(Math.round(DISPLAY * 0.12));

  const mask = useMemo(() => new FilterMask(DISPLAY), []);

  // Display params live in a ref so slider moves don't restart the loop. The
  // mask (and its version) are read live from the closure each frame instead,
  // so edits take effect without forcing a React re-render.
  const paramsRef = useRef({ logGain, colormap, filterEnabled });
  paramsRef.current = { logGain, colormap, filterEnabled };

  useEffect(() => {
    if (status !== "ready") return;
    if (!caps.webgl2 || !caps.colorBufferFloat) return;
    const video = videoRef.current;
    if (!video) return;

    let pipeline: Pipeline;
    try {
      pipeline = new Pipeline(fftSize);
    } catch (err) {
      setRenderError(err instanceof Error ? err.message : String(err));
      return;
    }

    let raf = 0;
    let frames = 0;
    let lastFpsT = performance.now();

    const loop = () => {
      const p = paramsRef.current;
      const params: PipelineParams = {
        logGain: p.logGain,
        colormap: p.colormap,
        filterEnabled: p.filterEnabled,
        mask: mask.canvas,
        maskVersion: mask.version,
      };
      pipeline.render(
        video,
        {
          grayscale: grayRef.current?.getContext("2d") ?? null,
          spectrum: spectrumRef.current?.getContext("2d") ?? null,
          reconstruction: reconRef.current?.getContext("2d") ?? null,
          roundtrip: roundtripRef.current?.getContext("2d") ?? null,
        },
        params,
      );

      frames++;
      const now = performance.now();
      if (now - lastFpsT >= 500) {
        setFps(Math.round((frames * 1000) / (now - lastFpsT)));
        frames = 0;
        lastFpsT = now;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      pipeline.dispose();
    };
  }, [status, caps, videoRef, fftSize, mask]);

  const ready = caps.webgl2 && caps.colorBufferFloat;

  // --- Mask drawing -----------------------------------------------------

  const redrawOverlay = () => {
    const ctx = overlayRef.current?.getContext("2d");
    if (ctx) mask.renderOverlay(ctx);
  };

  // Render the (empty) overlay once both the mask and canvas are present.
  useEffect(() => {
    if (ready) redrawOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  const drawing = useRef(false);
  const lastPt = useRef<{ x: number; y: number } | null>(null);
  // Mode for the in-progress stroke: right button erases (opposite of brush).
  const strokeMode = useRef<BrushMode>("block");

  const opposite = (m: BrushMode): BrushMode => (m === "block" ? "pass" : "block");

  const toCanvasCoords = (e: PointerEvent<HTMLCanvasElement>) => {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const updateCursor = (e: PointerEvent<HTMLCanvasElement>) => {
    const ring = cursorRef.current;
    if (!ring) return;
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const scale = rect.width / canvas.width;
    const d = brushSize * 2 * scale;
    ring.style.width = `${d}px`;
    ring.style.height = `${d}px`;
    ring.style.left = `${e.clientX - rect.left}px`;
    ring.style.top = `${e.clientY - rect.top}px`;
  };

  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drawing.current = true;
    strokeMode.current = e.button === 2 ? opposite(brushMode) : brushMode;
    const p = toCanvasCoords(e);
    lastPt.current = p;
    mask.point(p.x, p.y, brushSize, strokeMode.current);
    redrawOverlay();
  };

  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    updateCursor(e);
    if (!drawing.current) return;
    const p = toCanvasCoords(e);
    const last = lastPt.current ?? p;
    mask.stroke(last.x, last.y, p.x, p.y, brushSize, strokeMode.current);
    lastPt.current = p;
    redrawOverlay();
  };

  const endStroke = (e: PointerEvent<HTMLCanvasElement>) => {
    drawing.current = false;
    lastPt.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  const onPointerEnter = (e: PointerEvent<HTMLCanvasElement>) => {
    if (cursorRef.current) cursorRef.current.style.display = "block";
    updateCursor(e);
  };

  const onPointerLeave = (e: PointerEvent<HTMLCanvasElement>) => {
    if (cursorRef.current) cursorRef.current.style.display = "none";
    endStroke(e);
  };

  const preset = (fn: () => void) => () => {
    fn();
    redrawOverlay();
  };

  return (
    <div className="app">
      <h1>FFT Webcam</h1>
      <CapabilityBanner
        caps={caps}
        webcamStatus={status}
        webcamError={error}
        renderError={renderError}
      />

      {/* Hidden source video; the canvases are the actual output. */}
      <video ref={videoRef} playsInline muted />

      <div className="panes">
        <div className="pane">
          <h2>Input</h2>
          <canvas ref={grayRef} width={DISPLAY} height={DISPLAY} />
        </div>
        <div className="pane">
          <h2>Magnitude spectrum — draw to filter</h2>
          <div className="spectrum-stack">
            <canvas ref={spectrumRef} width={DISPLAY} height={DISPLAY} />
            <canvas
              ref={overlayRef}
              width={DISPLAY}
              height={DISPLAY}
              className="overlay-canvas"
              onPointerDown={ready ? onPointerDown : undefined}
              onPointerMove={ready ? onPointerMove : undefined}
              onPointerUp={endStroke}
              onPointerCancel={endStroke}
              onPointerEnter={ready ? onPointerEnter : undefined}
              onPointerLeave={onPointerLeave}
              onContextMenu={(e) => e.preventDefault()}
            />
            <div ref={cursorRef} className="cursor-ring" />
          </div>
        </div>
        {showRecon && (
          <div className="pane">
            <h2>Reconstruction (inverse FFT)</h2>
            <canvas ref={reconRef} width={DISPLAY} height={DISPLAY} />
          </div>
        )}
        {showRoundtrip && (
          <div className="pane">
            <h2>Round-trip spectrum (FFT of reconstruction)</h2>
            <canvas ref={roundtripRef} width={DISPLAY} height={DISPLAY} />
          </div>
        )}
      </div>

      {ready && (
        <>
          <div className="controls">
            <label>
              FFT size:
              <select
                value={fftSize}
                onChange={(e) => setFftSize(parseInt(e.target.value, 10))}
              >
                {FFT_SIZES.map((s) => (
                  <option key={s} value={s}>
                    {s}×{s}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Log gain: {logGain.toFixed(3)}
              <input
                type="range"
                min={0.01}
                max={0.3}
                step={0.005}
                value={logGain}
                onChange={(e) => setLogGain(parseFloat(e.target.value))}
              />
            </label>
            <label>
              Colormap:
              <select
                value={colormap}
                onChange={(e) => setColormap(parseInt(e.target.value, 10))}
              >
                <option value={1}>Viridis</option>
                <option value={0}>Grayscale</option>
              </select>
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={showRecon}
                onChange={(e) => setShowRecon(e.target.checked)}
              />
              Reconstruction
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={showRoundtrip}
                onChange={(e) => setShowRoundtrip(e.target.checked)}
              />
              Round-trip
            </label>
            <span className="fps">{fps} fps</span>
          </div>

          <div className="controls filter-controls">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={filterEnabled}
                onChange={(e) => setFilterEnabled(e.target.checked)}
              />
              Apply filter
            </label>
            <div className="brush-modes">
              <button
                className={brushMode === "block" ? "active" : ""}
                onClick={() => setBrushMode("block")}
              >
                Block brush
              </button>
              <button
                className={brushMode === "pass" ? "active" : ""}
                onClick={() => setBrushMode("pass")}
              >
                Pass brush
              </button>
            </div>
            <label>
              Brush size: {brushSize}px
              <input
                type="range"
                min={4}
                max={120}
                step={1}
                value={brushSize}
                onChange={(e) => setBrushSize(parseInt(e.target.value, 10))}
              />
            </label>
            <label>
              Preset radius: {cutoff}px
              <input
                type="range"
                min={8}
                max={DISPLAY / 2}
                step={1}
                value={cutoff}
                onChange={(e) => setCutoff(parseInt(e.target.value, 10))}
              />
            </label>
            <div className="presets">
              <button onClick={preset(() => mask.lowPass(cutoff))}>Low-pass</button>
              <button onClick={preset(() => mask.highPass(cutoff))}>High-pass</button>
              <button onClick={preset(() => mask.bandPass(cutoff, cutoff * 2))}>
                Band-pass
              </button>
              <button onClick={preset(() => mask.invert())}>Invert</button>
              <button onClick={preset(() => mask.allPass())}>All-pass</button>
              <button onClick={preset(() => mask.blockAll())}>Block all</button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function CapabilityBanner({
  caps,
  webcamStatus,
  webcamError,
  renderError,
}: {
  caps: ReturnType<typeof detectCapabilities>;
  webcamStatus: string;
  webcamError: string | null;
  renderError: string | null;
}) {
  if (!caps.webgl2) {
    return <div className="status error">WebGL2 is not available in this browser.</div>;
  }
  if (!caps.colorBufferFloat) {
    return (
      <div className="status error">
        EXT_color_buffer_float is unsupported — the GPU FFT needs float render
        targets. Try a desktop Chrome/Firefox/Edge.
      </div>
    );
  }
  if (renderError) {
    return <div className="status error">Renderer error: {renderError}</div>;
  }
  if (webcamStatus === "error") {
    return <div className="status error">Webcam error: {webcamError}</div>;
  }
  if (webcamStatus !== "ready") {
    return <div className="status warn">Waiting for webcam permission…</div>;
  }
  // Everything is working — no banner needed.
  return null;
}
