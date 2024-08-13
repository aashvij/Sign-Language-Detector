declare namespace ort {
    class InferenceSession {
        static create(path: string): Promise<InferenceSession>;
        run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
    }

    class Tensor {
        constructor(type: 'float32' | 'int32' | 'bool' | 'string', data: Float32Array | Int32Array | Uint8Array | string[], dims: number[]);
        data: Float32Array | Int32Array | Uint8Array | string[];
        dims: number[];
        type: 'float32' | 'int32' | 'bool' | 'string';
    }
}
