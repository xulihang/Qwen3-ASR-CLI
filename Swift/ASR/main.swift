//
//  main.swift
//  ASR
//
//  Created by 徐力航 on 2026/4/25.
//

import Foundation
import AVFAudio
import Qwen3ASR


// Load the model (downloads from HuggingFace on first use)
let model = try await Qwen3ASRModel.fromPretrained(modelId:"aufklarer/Qwen3-ASR-1.7B-MLX-8bit", offlineMode: true)
let fileURL = Bundle.main.url(forResource: "segment-00006", withExtension: "wav")!
let samples = try loadAudioSamples(from: fileURL)
// Transcribe an audio file
let result = model.transcribe(audio: samples,language: "ja")
print(result)
let alignModel = try await Qwen3ForcedAligner.fromPretrained(modelId:"aufklarer/Qwen3-ForcedAligner-0.6B-8bit",offlineMode:true);
let aligned = alignModel.align(audio: samples, text: result, language:"ja")
for word in aligned {
    print(word.startTime)
    print(word.endTime)
    print(word.text)
}
           
func loadAudioSamples(from url: URL, targetSampleRate: Double = 16000) throws -> [Float] {
    // 1. 打开音频文件
    let audioFile = try AVAudioFile(forReading: url)
    
    // 2. 检查并转换采样率
    guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                     sampleRate: targetSampleRate,
                                     channels: 1,  // 转换为单声道
                                     interleaved: false) else {
        throw NSError(domain: "AudioError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to create format"])
    }
    
    // 如果文件采样率不是 targetSampleRate，进行重采样
    let processingFormat: AVAudioFormat
    if audioFile.fileFormat.sampleRate != targetSampleRate {
        processingFormat = format
    } else {
        processingFormat = audioFile.processingFormat
    }
    
    // 3. 计算缓冲区容量 (假设最大处理60秒音频，可调整)
    let capacityFrames = AVAudioFrameCount(targetSampleRate * 60)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: processingFormat, frameCapacity: capacityFrames) else {
        throw NSError(domain: "AudioError", code: -2, userInfo: [NSLocalizedDescriptionKey: "Unable to create buffer"])
    }
    
    // 4. 读取音频数据到缓冲区
    try audioFile.read(into: buffer)
    
    // 5. 如果进行了重采样，需要将数据转换到目标格式
    let finalBuffer: AVAudioPCMBuffer
    if audioFile.fileFormat.sampleRate != targetSampleRate {
        guard let resampler = AVAudioConverter(from: audioFile.processingFormat, to: format),
              let resampledBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: buffer.frameLength) else {
            throw NSError(domain: "AudioError", code: -3, userInfo: [NSLocalizedDescriptionKey: "Unable to create resampler"])
        }
        
        var error: NSError?
        resampler.convert(to: resampledBuffer, error: &error) { inNumPackets, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }
        if let error = error { throw error }
        finalBuffer = resampledBuffer
    } else {
        finalBuffer = buffer
    }
    
    // 6. 提取 Float 数组
    guard let channelData = finalBuffer.floatChannelData else {
        throw NSError(domain: "AudioError", code: -4, userInfo: [NSLocalizedDescriptionKey: "No audio data"])
    }
    
    let frameLength = Int(finalBuffer.frameLength)
    let samples = Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
    return samples
}
