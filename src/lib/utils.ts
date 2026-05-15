import { pipeline, type TextGenerationPipeline, env } from '@huggingface/transformers';

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

export async function loadModel(name: string = "LiquidAI/LFM2.5-1.2B-Instruct-ONNX") {
  let device = await checkWebGPUSupport() ? 'webgpu' : 'auto';

  // Additional check for safari because it reports webgpu capability
  // and then fails to load it anyway.
  if (isSafari()) {
    device = 'auto';
  }

  console.info('Selected device:', device);

  return await pipeline("text-generation", name, {
    device: device,
    dtype: 'q4',
  }) as TextGenerationPipeline;
}
