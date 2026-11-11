/**
 * React Native entry: onnxruntime-react-native. The app supplies model storage (e.g. react-native-fs or Expo
 * FileSystem behind `ModelStorage`) and the microphone (e.g. react-native-live-audio-stream), pushing PCM into
 * `session.pushAudio` at whatever rate it records; the listener resamples.
 */
import * as ort from "onnxruntime-react-native";
import type { LoadedModel } from "./model.js";
import { fetchModel } from "./models.js";
import type { ModelRegistry, ModelStorage, TierName } from "./models.js";
import type { OrtModule } from "./ort.js";
import { ortBackend } from "./ort.js";
import { MODEL_REGISTRY } from "./registry.js";

export * from "./index.js";

export interface NativeModelOptions {
  tier?: TierName;
  registry?: ModelRegistry;
  storage: ModelStorage;
  /** Path the storage keeps the model at: onnxruntime-react-native loads files faster than bytes across the bridge. */
  pathOf?: (key: string) => string;
  /** e.g. ["nnapi", "cpu"] on Android, ["coreml", "cpu"] on iOS. Default CPU. */
  providers?: string[];
  threads?: number;
  allowOverBudget?: boolean;
  onProgress?: (loaded: number, total: number) => void;
}

export async function loadNativeModel(o: NativeModelOptions): Promise<LoadedModel> {
  const { tier, bytes } = await fetchModel({
    tier: o.tier ?? "full",
    registry: o.registry ?? MODEL_REGISTRY,
    storage: o.storage,
    allowOverBudget: o.allowOverBudget,
    onProgress: o.onProgress,
  });
  const key = `${tier.id}@${tier.version}/${tier.manifest.file}`;
  const backend = await ortBackend(ort as unknown as OrtModule, o.pathOf ? o.pathOf(key) : bytes, "onnxruntime-react-native", {
    executionProviders: o.providers ?? ["cpu"],
    intraOpNumThreads: o.threads ?? 1,
  });
  return { manifest: tier.manifest, backend, supported: tier.supported };
}
