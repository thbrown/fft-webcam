import { useEffect, useRef, useState, type RefObject } from "react";

export type WebcamStatus = "idle" | "requesting" | "ready" | "error";

export interface WebcamState {
  videoRef: RefObject<HTMLVideoElement | null>;
  status: WebcamStatus;
  error: string | null;
}

/**
 * Requests the webcam and pipes it into a hidden <video> element. The element
 * is what the WebGL pipeline uploads as a texture each frame.
 */
export function useWebcam(): WebcamState {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [status, setStatus] = useState<WebcamStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;

    async function start() {
      setStatus("requesting");
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const video = videoRef.current;
        if (!video) return;
        video.srcObject = stream;
        await video.play();
        setStatus("ready");
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setStatus("error");
      }
    }

    start();

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return { videoRef, status, error };
}
