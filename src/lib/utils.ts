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

  const model = "LiquidAI/LFM2-8B-A1B-ONNX";
  console.info('Loading model:', model);

  return await pipeline("text-generation", model, {
    device: device,
    dtype: 'q4',
    progress_callback,
  }) as TextGenerationPipeline;
}
