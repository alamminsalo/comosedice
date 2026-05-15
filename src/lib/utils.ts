import { pipeline, type TextGenerationPipeline, type ProgressCallback, env } from '@huggingface/transformers';

async function selectDevice(): Promise<string> {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      return 'webgpu';
    }
  } catch (e) {
    console.warn(e);
  }
  return 'auto';
}

export function isSafari(): boolean {
  const ua = navigator.userAgent;
  // Chrome/Edge/Brave include "Chrome" and "Safari". 
  // True Safari has "Safari" but NOT "Chrome".
  return ua.includes('Safari') && !ua.includes('Chrome') && !ua.includes('Chromium');
}

export async function loadModel(progress_callback: undefined | ProgressCallback = undefined) {
  let device = await selectDevice();
  //if (isSafari()) {
  //  device = 'auto';
  //}
  console.info('Selected device:', device);

  const model = "LiquidAI/LFM2-8B-A1B-ONNX";
  console.info('Loading model:', model);

  return await pipeline("text-generation", model, {
    device: device,
    dtype: 'q4',
    progress_callback,
  }) as TextGenerationPipeline;
}

// utils.js
export function parseMarkdown(text: string) {
  if (!text) return "";

  return text
    .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>") // Bold: **text**
    .replace(/\*(.*?)\*/g, "<em>$1</em>")             // Italics: *text*
    .replace(/\n/g, "<br />");                        // Newlines: \n
}
