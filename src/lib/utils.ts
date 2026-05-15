import { pipeline, type TextGenerationPipeline, type ProgressCallback, env } from '@huggingface/transformers';

export async function checkWebGPUSupport(): Promise<boolean> {
  // 1. Check if the browser API exists
  if (!navigator.gpu) return false;

  try {
    // 2. Check if an adapter (physical GPU) is accessible
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;

    // 3. Optional: Configure Transformers.js to prioritize WebGPU
    // This helps the 'auto' device selection be more aggressive
    env.allowLocalModels = false; // Example config

    return true;
  } catch (e) {
    return false;
  }
}

function isMobileDevice() {
  // 1. Check for modern Client Hints (Chrome/Android support)
  if (navigator.userAgentData) {
    return navigator.userAgentData.mobile;
  }

  // 2. Check for Touch + Screen Size (Reliable fallback for iOS Safari)
  const hasTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
  const isSmallScreen = window.matchMedia("(max-width: 768px)").matches;

  // 3. Logic: If it has touch and a small screen, it's a phone.
  // If it has touch and a large screen, it's likely an iPad (which also has RAM limits).
  return hasTouch && (isSmallScreen || /iPad|iPhone|iPod/.test(navigator.platform));
}

export function isSafari(): boolean {
  const ua = navigator.userAgent;
  // Chrome/Edge/Brave include "Chrome" and "Safari". 
  // True Safari has "Safari" but NOT "Chrome".
  return ua.includes('Safari') && !ua.includes('Chrome') && !ua.includes('Chromium');
}

export async function loadModel(progress_callback: undefined | ProgressCallback = undefined) {
  let device = await checkWebGPUSupport() ? 'webgpu' : 'auto';
  if (isSafari()) {
    device = 'auto';
  }
  console.info('Selected device:', device);

  const model = isMobileDevice() ? 'LiquidAI/LFM2.5-350M-ONNX' : "LiquidAI/LFM2.5-1.2B-Instruct-ONNX";
  console.info('Loading model:', model);

  return await pipeline("text-generation", model, {
    device: device,
    dtype: 'q4',
    progress_callback,
  }) as TextGenerationPipeline;
}
