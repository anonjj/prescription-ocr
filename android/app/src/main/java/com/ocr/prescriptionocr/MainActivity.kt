package com.ocr.prescriptionocr

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import coil.load
import com.google.android.material.snackbar.Snackbar
import com.ocr.prescriptionocr.databinding.ActivityMainBinding
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors

/**
 * Main screen of the Prescription OCR app — fully offline.
 *
 * Flow:
 *  1. User picks an image from gallery OR takes a photo with the camera.
 *  2. Image is shown in the preview pane.
 *  3. User taps "Recognise" — image is preprocessed and fed to the on-device
 *     PyTorch Mobile model (model.ptl stored in assets/).
 *  4. CTC greedy decoding produces the recognised text, shown in the result card.
 *
 * No internet connection required.
 */
class MainActivity : AppCompatActivity() {

    // ── Character set (must match config.py CHARS exactly) ───────────────────
    // Index 0 is the CTC blank token; visible chars start at index 1.
    companion object {
        private const val CHARS =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,-/'()+"

        // Input dimensions the model was trained on (config.py IMG_HEIGHT / IMG_WIDTH)
        private const val IMG_H = 64
        private const val IMG_W = 256
    }

    // ── View binding ─────────────────────────────────────────────────────────
    private lateinit var binding: ActivityMainBinding

    // Loaded once on first use (lazy) — loading from assets takes ~0.5 s
    private val model: Module by lazy { loadModel() }

    // Background thread for inference (keeps UI responsive)
    private val inferenceExecutor = Executors.newSingleThreadExecutor()

    // URI of the current photo taken by the camera
    private var cameraImageUri: Uri? = null

    // URI of whichever image the user has selected (gallery or camera)
    private var selectedImageUri: Uri? = null

    // ── Activity Result Launchers ─────────────────────────────────────────────

    /** Opens the system gallery picker. */
    private val galleryLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            if (uri != null) {
                selectedImageUri = uri
                showImagePreview(uri)
            }
        }

    /** Launches the camera to capture a photo. */
    private val cameraLauncher =
        registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
            if (success && cameraImageUri != null) {
                selectedImageUri = cameraImageUri
                showImagePreview(cameraImageUri!!)
            }
        }

    /** Asks for camera permission before opening the camera. */
    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) openCamera()
            else showSnackbar("Camera permission is required to take a photo.")
        }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setupButtons()
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceExecutor.shutdown()
    }

    // ── UI ────────────────────────────────────────────────────────────────────

    private fun setupButtons() {
        binding.btnGallery.setOnClickListener {
            galleryLauncher.launch("image/*")
        }
        binding.btnCamera.setOnClickListener {
            if (hasCameraPermission()) openCamera()
            else cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
        binding.btnRecognise.setOnClickListener {
            val uri = selectedImageUri
            if (uri == null) showSnackbar("Please select or capture an image first.")
            else runInference(uri)
        }
    }

    private fun showImagePreview(uri: Uri) {
        binding.imagePreview.load(uri) { crossfade(true) }
        binding.tvResult.text = ""
        binding.cardResult.visibility = View.GONE
        binding.btnRecognise.isEnabled = true
    }

    // ── Camera ────────────────────────────────────────────────────────────────

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    private fun openCamera() {
        val photoFile = createTempImageFile()
        cameraImageUri = FileProvider.getUriForFile(
            this, "${packageName}.fileprovider", photoFile
        )
        cameraLauncher.launch(cameraImageUri)
    }

    /** Creates a uniquely named JPEG file in the app's private Pictures folder. */
    private fun createTempImageFile(): File {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return File.createTempFile("OCR_${ts}_", ".jpg",
            getExternalFilesDir(Environment.DIRECTORY_PICTURES))
    }

    // ── Model Loading ─────────────────────────────────────────────────────────

    /**
     * Copies model.ptl from assets to the app's internal files directory
     * (PyTorch Mobile requires a file path, not an InputStream) then loads it.
     */
    private fun loadModel(): Module {
        val modelFile = File(filesDir, "model.ptl")
        if (!modelFile.exists()) {
            assets.open("model.ptl").use { input ->
                FileOutputStream(modelFile).use { output -> input.copyTo(output) }
            }
        }
        return LiteModuleLoader.load(modelFile.absolutePath)
    }

    // ── Inference Pipeline ────────────────────────────────────────────────────

    /**
     * Runs the full OCR pipeline on a background thread:
     *   URI → Bitmap → grayscale → resize/pad → float tensor → model → CTC decode
     */
    private fun runInference(uri: Uri) {
        binding.btnRecognise.isEnabled = false
        binding.progressBar.visibility = View.VISIBLE
        binding.cardResult.visibility = View.GONE

        inferenceExecutor.execute {
            try {
                // 1. Decode URI to Bitmap
                val bitmap = uriToBitmap(uri)

                // 2. Preprocess: grayscale → resize with aspect-ratio padding → normalise
                val floatArray = preprocessBitmap(bitmap)

                // 3. Create input tensor shape (1, 1, H, W)
                val inputTensor = Tensor.fromBlob(
                    floatArray,
                    longArrayOf(1L, 1L, IMG_H.toLong(), IMG_W.toLong())
                )

                // 4. Forward pass through the CRNN
                val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()

                // 5. Decode the CTC output to a string
                val text = ctcDecode(outputTensor)

                runOnUiThread { showResult(text) }

            } catch (e: Exception) {
                runOnUiThread { showError(e.message ?: "Unknown error") }
            }
        }
    }

    // ── Image Preprocessing ───────────────────────────────────────────────────

    /** Decodes a content URI to a Bitmap. */
    private fun uriToBitmap(uri: Uri): Bitmap {
        val stream = contentResolver.openInputStream(uri)
            ?: throw IllegalArgumentException("Cannot open URI: $uri")
        return BitmapFactory.decodeStream(stream)
            ?: throw IllegalArgumentException("Cannot decode image")
    }

    /**
     * Prepares a Bitmap for the CRNN model — mirrors the Python preprocessing pipeline:
     *  1. Convert to grayscale
     *  2. Binarise with Otsu's threshold (matches adaptive_threshold in transforms.py)
     *  3. Scale to IMG_H height, preserving aspect ratio
     *  4. Pad width to IMG_W with white (255)
     *  5. Normalise pixel values to [0, 1]
     *
     * Returns a flat float array of length IMG_H * IMG_W.
     */
    private fun preprocessBitmap(src: Bitmap): FloatArray {
        // Step 1 — Grayscale: read luma value for every pixel
        val w = src.width
        val h = src.height
        val srcPixels = IntArray(w * h)
        src.getPixels(srcPixels, 0, w, 0, 0, w, h)

        val gray = IntArray(w * h) { i ->
            val p = srcPixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8)  and 0xFF
            val b =  p         and 0xFF
            // Standard luminance weights
            (0.299 * r + 0.587 * g + 0.114 * b).toInt()
        }

        // Step 2 — Otsu's threshold: compute histogram, find best split point
        val histogram = IntArray(256)
        for (v in gray) histogram[v]++

        val total = w * h
        var sumAll = 0L
        for (i in 0..255) sumAll += i * histogram[i]

        var sumB = 0L
        var wB = 0
        var threshold = 128   // fallback
        var maxVariance = 0.0

        for (t in 0..255) {
            wB += histogram[t]
            if (wB == 0) continue
            val wF = total - wB
            if (wF == 0) break

            sumB += t * histogram[t]
            val meanB = sumB.toDouble() / wB
            val meanF = (sumAll - sumB).toDouble() / wF
            val variance = wB.toDouble() * wF * (meanB - meanF) * (meanB - meanF)
            if (variance > maxVariance) { maxVariance = variance; threshold = t }
        }

        // Apply threshold: dark pixels (text) → 0, light pixels (background) → 255
        val binary = IntArray(w * h) { i -> if (gray[i] < threshold) 0 else 255 }

        // Step 3 — Build a greyscale Bitmap from the binary array, scale to IMG_H
        val binaryBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val binaryArgb = IntArray(w * h) { i ->
            val v = binary[i]; (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        binaryBmp.setPixels(binaryArgb, 0, w, 0, 0, w, h)

        val scale  = IMG_H.toFloat() / h
        val scaledW = minOf((w * scale).toInt(), IMG_W)
        val scaled  = Bitmap.createScaledBitmap(binaryBmp, scaledW, IMG_H, true)

        // Step 4 — Pad to IMG_W × IMG_H with white
        val padded    = Bitmap.createBitmap(IMG_W, IMG_H, Bitmap.Config.ARGB_8888)
        val padCanvas = android.graphics.Canvas(padded)
        padCanvas.drawColor(Color.WHITE)
        padCanvas.drawBitmap(scaled, 0f, 0f, null)

        // Step 5 — Normalise to [0, 1]
        val finalPixels = IntArray(IMG_W * IMG_H)
        padded.getPixels(finalPixels, 0, IMG_W, 0, 0, IMG_W, IMG_H)

        return FloatArray(IMG_W * IMG_H) { i ->
            ((finalPixels[i] shr 16) and 0xFF) / 255.0f
        }
    }

    // ── CTC Decoding ──────────────────────────────────────────────────────────

    /**
     * Greedy CTC decoder — mirrors model/utils.py:decode_prediction().
     *
     * The model output tensor has shape (seq_len, 1, num_classes) stored row-major.
     * At each time step we pick the class with the highest log-probability, then:
     *   - skip index 0 (blank token)
     *   - collapse consecutive repeated indices
     */
    private fun ctcDecode(tensor: Tensor): String {
        val data      = tensor.dataAsFloatArray   // flat array, length = seqLen * numClasses
        val shape     = tensor.shape()            // [seqLen, 1, numClasses]
        val seqLen    = shape[0].toInt()
        val numClasses = shape[2].toInt()

        val sb = StringBuilder()
        var prevIdx = 0

        for (t in 0 until seqLen) {
            // Find argmax over the numClasses dimension at time step t
            var maxIdx = 0
            var maxVal = Float.NEGATIVE_INFINITY
            for (c in 0 until numClasses) {
                val v = data[t * numClasses + c]
                if (v > maxVal) { maxVal = v; maxIdx = c }
            }

            // Append character only if not blank (0) and not a repeat
            if (maxIdx != 0 && maxIdx != prevIdx) {
                val charIdx = maxIdx - 1   // shift by 1 because index 0 is blank
                if (charIdx < CHARS.length) sb.append(CHARS[charIdx])
            }
            prevIdx = maxIdx
        }

        return sb.toString()
    }

    // ── Result Display ────────────────────────────────────────────────────────

    private fun showResult(text: String) {
        binding.progressBar.visibility = View.GONE
        binding.btnRecognise.isEnabled = true

        val display = if (text.isBlank()) "No text recognised. Try a clearer image."
                      else "Recognised text:\n\n$text"

        binding.tvResult.text = display
        binding.cardResult.visibility = View.VISIBLE
    }

    private fun showError(message: String) {
        binding.progressBar.visibility = View.GONE
        binding.btnRecognise.isEnabled = true
        binding.tvResult.text = "Error: $message"
        binding.cardResult.visibility = View.VISIBLE
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun showSnackbar(message: String) =
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG).show()
}
