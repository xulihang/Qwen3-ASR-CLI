//
//  main.swift
//  ASR
//
//  Created by 徐力航 on 2026/4/25.
//  Modified to support command-line arguments (top-level async entry).
//

import Foundation
import AVFAudio
import Qwen3ASR

// MARK: - Command Line Argument Parsing

struct CommandLineOptions {
    var asrModelId: String = "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
    var asrCacheDir: URL = URL(fileURLWithPath: "models/Qwen3-ASR-1.7B-MLX-8bit")
    var alignerModelId: String = "aufklarer/Qwen3-ForcedAligner-0.6B-8bit"
    var alignerCacheDir: URL = URL(fileURLWithPath: "models/Qwen3-ForcedAligner-0.6B-8bit")
    var language: String? = nil
    var mode: String = ""               // "transcribe" or "align"
    var wavListPath: String = ""
    var textFilePath: String = ""       // Required for align mode
    var offline: Bool = true
    var outputDir: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

    func validate() throws {
        if mode.isEmpty {
            throw NSError(domain: "ArgumentError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing --mode (transcribe/align)"])
        }
        if mode != "transcribe" && mode != "align" {
            throw NSError(domain: "ArgumentError", code: 3, userInfo: [NSLocalizedDescriptionKey: "Invalid --mode, must be 'transcribe' or 'align'"])
        }
        if wavListPath.isEmpty {
            throw NSError(domain: "ArgumentError", code: 4, userInfo: [NSLocalizedDescriptionKey: "Missing --wav-list"])
        }
        if mode == "align" && textFilePath.isEmpty {
            throw NSError(domain: "ArgumentError", code: 5, userInfo: [NSLocalizedDescriptionKey: "Align mode requires --text-file"])
        }
    }
}

func parseArguments() throws -> CommandLineOptions {
    var opts = CommandLineOptions()
    let args = CommandLine.arguments
    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "--asr-model-id":
            i += 1
            if i < args.count { opts.asrModelId = args[i] }
        case "--asr-cache-dir":
            i += 1
            if i < args.count { opts.asrCacheDir = URL(fileURLWithPath: args[i]) }
        case "--aligner-model-id":
            i += 1
            if i < args.count { opts.alignerModelId = args[i] }
        case "--aligner-cache-dir":
            i += 1
            if i < args.count { opts.alignerCacheDir = URL(fileURLWithPath: args[i]) }
        case "--language":
            i += 1
            if i < args.count { opts.language = args[i] }
        case "--mode":
            i += 1
            if i < args.count { opts.mode = args[i] }
        case "--wav-list":
            i += 1
            if i < args.count { opts.wavListPath = args[i] }
        case "--text-file":
            i += 1
            if i < args.count { opts.textFilePath = args[i] }
        case "--offline":
            opts.offline = true
        case "--output-dir":
            i += 1
            if i < args.count { opts.outputDir = URL(fileURLWithPath: args[i]) }
        default:
            fputs("Warning: unknown argument \(arg)\n", stderr)
        }
        i += 1
    }
    try opts.validate()
    return opts
}

// MARK: - Helper Functions

func loadAudioSamples(from url: URL, targetSampleRate: Double = 16000) throws -> [Float] {
    let audioFile = try AVAudioFile(forReading: url)
    guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                     sampleRate: targetSampleRate,
                                     channels: 1,
                                     interleaved: false) else {
        throw NSError(domain: "AudioError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to create format"])
    }
    let processingFormat: AVAudioFormat
    if audioFile.fileFormat.sampleRate != targetSampleRate {
        processingFormat = format
    } else {
        processingFormat = audioFile.processingFormat
    }
    let capacityFrames = AVAudioFrameCount(targetSampleRate * 60)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: processingFormat, frameCapacity: capacityFrames) else {
        throw NSError(domain: "AudioError", code: -2, userInfo: [NSLocalizedDescriptionKey: "Unable to create buffer"])
    }
    try audioFile.read(into: buffer)
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
    guard let channelData = finalBuffer.floatChannelData else {
        throw NSError(domain: "AudioError", code: -4, userInfo: [NSLocalizedDescriptionKey: "No audio data"])
    }
    let frameLength = Int(finalBuffer.frameLength)
    let samples = Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
    return samples
}

func readLines(from path: String) throws -> [String] {
    let content = try String(contentsOfFile: path, encoding: .utf8)
    return content.components(separatedBy: .newlines)
        .map { $0.trimmingCharacters(in: .whitespaces) }
        .filter { !$0.isEmpty && !$0.hasPrefix("#") }
}

// MARK: - Main Entry Point

func run() async {
    do {
        let opts = try parseArguments()

        try FileManager.default.createDirectory(at: opts.outputDir, withIntermediateDirectories: true)

        let wavPaths = try readLines(from: opts.wavListPath)
        guard !wavPaths.isEmpty else {
            fputs("No wav files found in list.\n", stderr)
            return
        }

        var textLines: [String]? = nil
        if opts.mode == "align" {
            textLines = try readLines(from: opts.textFilePath)
            guard textLines!.count == wavPaths.count else {
                fputs("Text file line count (\(textLines!.count)) does not match wav file count (\(wavPaths.count)).\n", stderr)
                return
            }
        }

        for (idx, wavPath) in wavPaths.enumerated() {
            let wavURL = URL(fileURLWithPath: wavPath)
            let fileName = wavURL.deletingPathExtension().lastPathComponent + "-" + opts.mode
            let outputURL = opts.outputDir.appendingPathComponent(fileName).appendingPathExtension("json")

            fputs("Processing \(wavPath)...\n", stderr)
            do {
                let samples = try loadAudioSamples(from: wavURL)
                var resultDict: [String: Any] = [:]
                resultDict["file"] = wavPath

                if opts.mode == "transcribe" {
                    fputs("Loading ASR model...\n", stderr)

                    let asrModel = try await Qwen3ASRModel.fromPretrained(
                        modelId: opts.asrModelId,
                        cacheDir: opts.asrCacheDir,
                        offlineMode: opts.offline
                    )
                    
                    let transcription = asrModel.transcribe(audio: samples, language: opts.language)
                    resultDict["transcription"] = transcription
                } else { // align
                    let text = textLines![idx]
                    fputs("Loading Aligner model...\n", stderr)
                    
                    var alignerModel: Qwen3ForcedAligner? = nil
                    alignerModel = try await Qwen3ForcedAligner.fromPretrained(
                        modelId: opts.alignerModelId,
                        cacheDir: opts.alignerCacheDir,
                        offlineMode: opts.offline
                    )
                    
                    let alignment = alignerModel!.align(audio: samples, text: text, language: opts.language ?? "English")
                    let words = alignment.map { word in
                        return [
                            "start": word.startTime,
                            "end": word.endTime,
                            "text": word.text
                        ]
                    }
                    resultDict["reference_text"] = text
                    resultDict["alignment"] = words
                }

                let jsonData = try JSONSerialization.data(withJSONObject: resultDict, options: [.prettyPrinted, .withoutEscapingSlashes])
                try jsonData.write(to: outputURL)
                fputs("Saved result to \(outputURL.path)\n", stderr)
                fputs("Progress \(idx+1)/\(wavPaths.count)\n", stderr)
            } catch {
                fputs("Error processing \(wavPath): \(error.localizedDescription)\n", stderr)
                let errorDict: [String: Any] = [
                    "file": wavPath,
                    "error": error.localizedDescription
                ]
                if let jsonData = try? JSONSerialization.data(withJSONObject: errorDict, options: .prettyPrinted) {
                    try? jsonData.write(to: outputURL)
                }
            }
        }

        fputs("All done.\n", stderr)
    } catch {
        fputs("Fatal error: \(error.localizedDescription)\n", stderr)
        fputs("Usage:\n", stderr)
        fputs("  --asr-model-id <id>          (default: aufklarer/Qwen3-ASR-1.7B-MLX-8bit)\n", stderr)
        fputs("  --asr-cache-dir <dir>        (default: models/Qwen3-ASR-1.7B-MLX-8bit)\n", stderr)
        fputs("  --aligner-model-id <id>      (default: aufklarer/Qwen3-ForcedAligner-0.6B-8bit)\n", stderr)
        fputs("  --aligner-cache-dir <dir>    (default: models/Qwen3-ForcedAligner-0.6B-8bit)\n", stderr)
        fputs("  --language <lang>            (e.g., ja, zh, en)\n", stderr)
        fputs("  --mode <transcribe|align>    required\n", stderr)
        fputs("  --wav-list <file>            txt file with one wav path per line\n", stderr)
        fputs("  --text-file <file>           required for align mode, text per line matching wav list\n", stderr)
        fputs("  --offline                    use cached models only, no download\n", stderr)
        fputs("  --output-dir <dir>           default: current directory\n", stderr)
        exit(1)
    }
}

// Start the async task
Task {
    await run()
    exit(0)
}

// Keep the runloop alive for async operations
RunLoop.main.run()
