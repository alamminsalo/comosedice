<script lang="ts">
  import { onMount } from 'svelte';
  import { pipeline, TextStreamer, type TextGenerationPipeline } from '@huggingface/transformers';
  import { loadModel } from './utils.ts';

  let isLoading = true;
  let progress = 0;
  let model: TextGenerationPipeline | null = null;
  let streamer: TextStreamer | null = null;
  let isGenerating = false;

  let inputText = '';
  let outputBuf = '';
  let outputText = '';
  let errorText = '';

  async function init() {
    try {
      model = await loadModel((e) => {
        if (e.status == 'progress_total') {
          progress = Math.trunc(e.progress);
        }
      });
    } catch (e) {
      console.error(e);
      errorText = 'Loading error. Please try again.';
    } finally {
      isLoading = false;
    }

    // Streamer pushes text to buffer and buffer is drained one character at a time.
    // This creates a typewriter effect.
    streamer = new TextStreamer(model.tokenizer, {
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
    if (!model || !inputText.trim()) return;
    
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
      const output = await model(messages, {
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
    await init();
  });
</script>

<div class="">
  {#if errorText}
    <div class="relative flex items-center justify-center text-2xl text-red-500">
      {errorText}
    </div>
  {:else}
    {#if isLoading}
      <progress max="100" value={progress} class="w-full h-4 bg-white border border-white rounded-full overflow-hidden appearance-none
        [&::-webkit-progress-bar]:bg-white 
        [&::-webkit-progress-value]:bg-orange-600 
        [&::-moz-progress-bar]:bg-orange-600"/>
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
