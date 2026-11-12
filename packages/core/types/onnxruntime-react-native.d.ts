// Type-checks the React Native entry without installing onnxruntime-react-native (and react-native, its peer).
// It re-exports the same onnxruntime-common API as onnxruntime-node/web. Not shipped.
declare module "onnxruntime-react-native" {
  export * from "onnxruntime-common";
}
