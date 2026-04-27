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
    var mode: String = ""               // "transcribe", "align", or empty for both
    var wavListPath: String = ""        // For batch processing
    var wavFilePath: String = ""        // For single file processing
    var textFilePath: String = ""       // Required for align mode when using wav-list
    var text: String = ""               // Optional text for single file align mode
    var offline: Bool = true
    var outputDir: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    var outputFilePath: String = ""     // For single file output

    func validate() throws {
        // Check if we have either wav-list or wav-file
        let hasWavList = !wavListPath.isEmpty
        let hasWavFile = !wavFilePath.isEmpty
        
        if !hasWavList && !hasWavFile {
            throw NSError(domain: "ArgumentError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing --wav-list or --wav-file"])
        }
        
        if hasWavList && hasWavFile {
            throw NSError(domain: "ArgumentError", code: 3, userInfo: [NSLocalizedDescriptionKey: "Cannot use both --wav-list and --wav-file"])
        }
        
        // Validate mode if provided
        if !mode.isEmpty && mode != "transcribe" && mode != "align" {
            throw NSError(domain: "ArgumentError", code: 4, userInfo: [NSLocalizedDescriptionKey: "Invalid --mode, must be 'transcribe', 'align', or omitted for both"])
        }
        
        // For batch mode with align, need text file
        if hasWavList && mode == "align" && textFilePath.isEmpty {
            throw NSError(domain: "ArgumentError", code: 5, userInfo: [NSLocalizedDescriptionKey: "Align mode with --wav-list requires --text-file"])
        }
        
        // For single file align mode, text is optional (will use ASR transcription if not provided)
        // So no validation needed here
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
        case "--wav-file":
            i += 1
            if i < args.count { opts.wavFilePath = args[i] }
        case "--text-file":
            i += 1
            if i < args.count { opts.textFilePath = args[i] }
        case "--text":
            i += 1
            if i < args.count { opts.text = args[i] }
        case "--offline":
            opts.offline = true
        case "--output-dir":
            i += 1
            if i < args.count { opts.outputDir = URL(fileURLWithPath: args[i]) }
        case "--output-file":
            i += 1
            if i < args.count { opts.outputFilePath = args[i] }
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

// MARK: - Processing Functions

func processTranscribe(model: Qwen3ASRModel, samples: [Float], language: String?) async throws -> String {
    return model.transcribe(audio: samples, language: language)
}

func processAlign(model: Qwen3ForcedAligner, samples: [Float], text: String, language: String?) async throws -> [[String: Any]] {
    let alignment = model.align(audio: samples, text: text, language: language ?? "English")
    return alignment.map { word in
        return [
            "start": word.startTime,
            "end": word.endTime,
            "text": word.text
        ]
    }
}

func processSingleFile(wavURL: URL, outputURL: URL, opts: CommandLineOptions) async throws {
    fputs("Processing \(wavURL.path)...\n", stderr)
    
    let samples = try loadAudioSamples(from: wavURL)
    var resultDict: [String: Any] = [:]
    resultDict["file"] = wavURL.path
    
    // Determine what to do based on mode
    let shouldTranscribe = opts.mode.isEmpty || opts.mode == "transcribe"
    let shouldAlign = opts.mode.isEmpty || opts.mode == "align"
    
    var transcription: String? = nil
    var asrModel: Qwen3ASRModel? = nil
    
    // Load ASR model if needed for transcription or for generating text for alignment
    let needsAsrForAlignment = shouldAlign && opts.text.isEmpty
    let needsAsrModel = shouldTranscribe || needsAsrForAlignment
    
    if needsAsrModel {
        fputs("Loading ASR model...\n", stderr)
        asrModel = try await Qwen3ASRModel.fromPretrained(
            modelId: opts.asrModelId,
            cacheDir: opts.asrCacheDir,
            offlineMode: opts.offline
        )
    }
    
    if shouldTranscribe {
        fputs("Performing transcription...\n", stderr)
        transcription = try await processTranscribe(model: asrModel!, samples: samples, language: opts.language)
        resultDict["transcription"] = transcription!
        fputs("Transcription: \(transcription!)\n", stderr)
    }
    
    if shouldAlign {
        // Determine which text to use for alignment
        let alignmentText: String
        if !opts.text.isEmpty {
            alignmentText = opts.text
            fputs("Using provided text for alignment: \(alignmentText)\n", stderr)
        } else if let transText = transcription {
            alignmentText = transText
            fputs("Using ASR transcription for alignment: \(alignmentText)\n", stderr)
        } else if let asrModel = asrModel {
            // If we didn't transcribe but need text for alignment, transcribe now
            fputs("Transcribing for alignment...\n", stderr)
            alignmentText = try await processTranscribe(model: asrModel, samples: samples, language: opts.language)
            resultDict["transcription"] = alignmentText
            fputs("Transcription for alignment: \(alignmentText)\n", stderr)
        } else {
            throw NSError(domain: "ProcessError", code: -1, userInfo: [NSLocalizedDescriptionKey: "No text available for alignment and ASR model not loaded"])
        }
        
        fputs("Loading Aligner model...\n", stderr)
        let alignerModel = try await Qwen3ForcedAligner.fromPretrained(
            modelId: opts.alignerModelId,
            cacheDir: opts.alignerCacheDir,
            offlineMode: opts.offline
        )
        
        fputs("Performing alignment...\n", stderr)
        let alignment = try await processAlign(model: alignerModel, samples: samples, text: alignmentText, language: opts.language)
        resultDict["reference_text"] = alignmentText
        resultDict["alignment"] = alignment
        fputs("Alignment completed with \(alignment.count) words\n", stderr)
    }
    
    let jsonData = try JSONSerialization.data(withJSONObject: resultDict, options: [.prettyPrinted, .withoutEscapingSlashes])
    try jsonData.write(to: outputURL)
    fputs("Saved result to \(outputURL.path)\n", stderr)
}

func processBatch(wavPaths: [String], textLines: [String]?, opts: CommandLineOptions) async throws {
    // For batch mode, we need to handle models more efficiently
    // Load models once if possible
    let shouldTranscribe = opts.mode.isEmpty || opts.mode == "transcribe"
    let shouldAlign = opts.mode.isEmpty || opts.mode == "align"
    
    var asrModel: Qwen3ASRModel? = nil
    var alignerModel: Qwen3ForcedAligner? = nil
    
    if shouldTranscribe {
        fputs("Loading ASR model...\n", stderr)
        asrModel = try await Qwen3ASRModel.fromPretrained(
            modelId: opts.asrModelId,
            cacheDir: opts.asrCacheDir,
            offlineMode: opts.offline
        )
    }
    
    if shouldAlign {
        fputs("Loading Aligner model...\n", stderr)
        alignerModel = try await Qwen3ForcedAligner.fromPretrained(
            modelId: opts.alignerModelId,
            cacheDir: opts.alignerCacheDir,
            offlineMode: opts.offline
        )
    }
    
    for (idx, wavPath) in wavPaths.enumerated() {
        let wavURL = URL(fileURLWithPath: wavPath)
        let fileName = wavURL.deletingPathExtension().lastPathComponent + "-" + (opts.mode.isEmpty ? "both" : opts.mode)
        let outputURL = opts.outputDir.appendingPathComponent(fileName).appendingPathExtension("json")
        
        fputs("Processing \(wavPath)...\n", stderr)
        do {
            let samples = try loadAudioSamples(from: wavURL)
            var resultDict: [String: Any] = [:]
            resultDict["file"] = wavPath
            
            var transcription: String? = nil
            
            if shouldTranscribe, let model = asrModel {
                fputs("Performing transcription...\n", stderr)
                transcription = try await processTranscribe(model: model, samples: samples, language: opts.language)
                resultDict["transcription"] = transcription!
            }
            
            if shouldAlign, let model = alignerModel {
                // For batch mode with align, we need text from textLines or from transcription
                let alignmentText: String
                if let textLines = textLines, idx < textLines.count, !textLines[idx].isEmpty {
                    alignmentText = textLines[idx]
                    fputs("Using provided text for alignment\n", stderr)
                } else if let transText = transcription {
                    alignmentText = transText
                    fputs("Using ASR transcription for alignment\n", stderr)
                } else if let asrModel = asrModel {
                    // If we didn't transcribe but need text for alignment, transcribe now
                    fputs("Transcribing for alignment...\n", stderr)
                    alignmentText = try await processTranscribe(model: asrModel, samples: samples, language: opts.language)
                    resultDict["transcription"] = alignmentText
                } else {
                    throw NSError(domain: "ProcessError", code: -1, userInfo: [NSLocalizedDescriptionKey: "No text available for alignment"])
                }
                
                let alignment = try await processAlign(model: model, samples: samples, text: alignmentText, language: opts.language)
                resultDict["reference_text"] = alignmentText
                resultDict["alignment"] = alignment
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
}

// MARK: - Main Entry Point

func run() async {
    do {
        let opts = try parseArguments()
        
        // Check if we're in single file mode
        let isSingleFileMode = !opts.wavFilePath.isEmpty
        
        if isSingleFileMode {
            // Single file processing
            let wavURL = URL(fileURLWithPath: opts.wavFilePath)
            
            // Determine output URL
            let outputURL: URL
            if !opts.outputFilePath.isEmpty {
                outputURL = URL(fileURLWithPath: opts.outputFilePath)
                // Ensure parent directory exists
                try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            } else {
                try FileManager.default.createDirectory(at: opts.outputDir, withIntermediateDirectories: true)
                let fileName = wavURL.deletingPathExtension().lastPathComponent + "-" + (opts.mode.isEmpty ? "both" : opts.mode)
                outputURL = opts.outputDir.appendingPathComponent(fileName).appendingPathExtension("json")
            }
            
            try await processSingleFile(wavURL: wavURL, outputURL: outputURL, opts: opts)
        } else {
            // Batch processing
            try FileManager.default.createDirectory(at: opts.outputDir, withIntermediateDirectories: true)
            
            let wavPaths = try readLines(from: opts.wavListPath)
            guard !wavPaths.isEmpty else {
                fputs("No wav files found in list.\n", stderr)
                return
            }
            
            var textLines: [String]? = nil
            if !opts.textFilePath.isEmpty {
                textLines = try readLines(from: opts.textFilePath)
                if opts.mode == "align" && textLines!.count != wavPaths.count {
                    fputs("Warning: Text file line count (\(textLines!.count)) does not match wav file count (\(wavPaths.count)). Will use ASR for missing texts.\n", stderr)
                }
            } else if opts.mode == "align" {
                fputs("No text file provided, will use ASR transcription for alignment\n", stderr)
            }
            
            try await processBatch(wavPaths: wavPaths, textLines: textLines, opts: opts)
        }
        
        fputs("All done.\n", stderr)
    } catch {
        fputs("Fatal error: \(error.localizedDescription)\n", stderr)
        fputs("Usage:\n", stderr)
        fputs("  Batch mode (multiple files):\n", stderr)
        fputs("    --wav-list <file>            txt file with one wav path per line\n", stderr)
        fputs("    --text-file <file>           optional text file for alignment (will use ASR if missing)\n", stderr)
        fputs("  Single file mode:\n", stderr)
        fputs("    --wav-file <path>            single audio file to process\n", stderr)
        fputs("    --text <text>                optional reference text for align mode (will use ASR if not provided)\n", stderr)
        fputs("    --output-file <path>         output JSON file path (optional)\n", stderr)
        fputs("  General options:\n", stderr)
        fputs("    --asr-model-id <id>          (default: aufklarer/Qwen3-ASR-1.7B-MLX-8bit)\n", stderr)
        fputs("    --asr-cache-dir <dir>        (default: models/Qwen3-ASR-1.7B-MLX-8bit)\n", stderr)
        fputs("    --aligner-model-id <id>      (default: aufklarer/Qwen3-ForcedAligner-0.6B-8bit)\n", stderr)
        fputs("    --aligner-cache-dir <dir>    (default: models/Qwen3-ForcedAligner-0.6B-8bit)\n", stderr)
        fputs("    --language <lang>            (e.g., ja, zh, en)\n", stderr)
        fputs("    --mode <transcribe|align>    optional, if omitted will do both\n", stderr)
        fputs("    --offline                    use cached models only, no download\n", stderr)
        fputs("    --output-dir <dir>           default: current directory (batch mode)\n", stderr)
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
