"""Start Tinker sampling sessions for saved checkpoints.

Run this before eval so the OpenAI-compatible endpoint can serve them.
Keep this script running while eval is in progress.
"""

import asyncio
import signal

import tinker

CHECKPOINTS = {
    "batna": "tinker://e9eedaf2-b5af-5b1b-975f-ac96e8e8b666:train:0/sampler_weights/final",
    "surplus": "tinker://426c15f8-0db9-5010-879c-89b347d5f2a0:train:0/sampler_weights/final",
}


async def main():
    service = tinker.ServiceClient()
    samplers = {}

    for name, path in CHECKPOINTS.items():
        print(f"Starting sampler for {name}: {path}")
        sampler = await service.create_sampling_client_async(model_path=path)
        samplers[name] = sampler
        print(f"  {name} sampler ready")
        print(f"  attrs: {[a for a in dir(sampler) if not a.startswith('_')]}")
        for attr in dir(sampler):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(sampler, attr)
                if not callable(val):
                    print(f"  {attr} = {val}")
            except Exception:
                pass

    print(f"\n{len(samplers)} sampler(s) active. Run eval now.")
    print("Press Ctrl+C to shut down.\n")

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    print("\nShutting down samplers.")


if __name__ == "__main__":
    asyncio.run(main())
