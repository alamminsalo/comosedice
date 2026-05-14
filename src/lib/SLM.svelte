<script lang="ts">
  import { onMount } from 'svelte';
  import { pipeline, TextStreamer, type TextGenerationPipeline } from '@huggingface/transformers';

  let isLoading = true;
  let generator: TextGenerationPipeline | null = null;
  let streamer: TextStreamer | null = null;
  let isGenerating = false;

  let inputText = '';
  let outputBuf = '';
  let outputText = '';
  let errorText = '';

  async function loadModel() {
    try {
      generator = await pipeline("text-generation", "LiquidAI/LFM2.5-1.2B-Instruct-ONNX", {
        device: 'webgpu',
        dtype: 'q4',
      }) as TextGenerationPipeline;
    } catch (e) {
      try {
        generator = await pipeline("text-generation", "LiquidAI/LFM2.5-1.2B-Instruct-ONNX", {
          device: 'auto',
          dtype: 'q4',
        }) as TextGenerationPipeline;
      } catch (e) {
        console.error(e);
        errorText = 'Loading error. Please try again.';
      }
    } finally {
      isLoading = false;
    }

    // Streamer pushes text to buffer and buffer is drained one character at a time.
    // This creates a typewriter effect.
    streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      callback_function: (text) => { 
        outputBuf += text;
      },
    });
    setInterval(() => {
      if (outputBuf.length > 0) {
        outputText += outputBuf.slice(0,1)
        outputBuf = outputBuf.slice(1)
      }
    }, 24)
  }

  const generateResponse = async () => {
    if (!generator || !inputText.trim()) return;
    
    isGenerating = true;
    outputText = '';
    outputBuf = '';

    const messages = [
      { 
        role: "system", 
        content: "You are a spanish teacher evaluating student texts. Explain the given sentence. If there are mistakes, please let me know. Be concise and respond in english."
      },
      { 
        role: "user", 
        content: `Evaluate this text: "${inputText}"` 
      },
    ];

    try {
      // LFMs use chat templates for instructions
      const output = await generator(messages, {
        max_new_tokens: 200,
        do_sample: false,
        streamer,
      });
    } catch (e: any) {
      outputText = `Error: ${e.message}`;
    } finally {
      isGenerating = false;
    }
  };

  onMount(async () => {
    await loadModel();
  });
</script>

<div class="">
  {#if errorText}
    <div class="relative flex items-center justify-center text-2xl text-red-500">
      {errorText}
    </div>
  {:else}
    {#if isLoading}
      <div class="relative flex items-center justify-center text-2xl">
        <span class="loader"/>
      </div>
    {:else}
      <input 
          type="text"
          bind:value={inputText}
          disabled={isGenerating}
          onkeydown={(e) => e.key === 'Enter' && generateResponse()}
          class="w-full border-b-2 border-gray-200 bg-transparent py-4 text-5xl font-light tracking-tight transition-all duration-300 focus:border-orange-600 focus:outline-none disabled:opacity-50"
          placeholder="Escribe tu frase..."
          autofocus
        />

      {#if outputText}
        <div class="text-4xl">
          <p>{outputText}</p>
        </div>
      {/if}
    {/if}
  {/if}
</div>

<style>
</style>
