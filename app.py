from flask import Flask, render_template, request, redirect, url_for, flash, session
import cv2
import numpy as np
import os
import base64
from datetime import datetime
import logging 

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'ganti_dengan_kunci_rahasia_yang_kuat_dan_unik' # Ganti dengan kunci yang kuat dan acak
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Cek apakah file yang diunggah memiliki ekstensi yang diizinkan."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image(img, filename, folder):
    """Simpan gambar ke folder tertentu."""
    filepath = os.path.join(folder, filename)
    try:
        # Pastikan direktori ada sebelum menyimpan
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, img)
        return filepath
    except Exception as e:
        app.logger.error(f"Error saving image {filepath}: {e}")
        return None

def convert_to_base64(image_path):
    """Konversi file gambar ke base64 untuk ditampilkan di halaman web."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        app.logger.error(f"File not found for base64 conversion: {image_path}")
        return None
    except Exception as e:
        app.logger.error(f"Error converting image {image_path} to base64: {e}")
        return None

def apply_threshold(image, threshold_value=127):
    """Menerapkan thresholding pada gambar."""
    if image is None: return None
    if len(image.shape) > 2: # Berwarna
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: # Sudah grayscale
        gray = image.copy()
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def apply_edge_detection(image, method):
    """Menerapkan deteksi tepi pada gambar."""
    if image is None: return None
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method == 'sobel':
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        return np.uint8(np.clip(sobel, 0, 255))
    
    elif method == 'prewitt':
        kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewittx = cv2.filter2D(blurred, cv2.CV_64F, kernelx)
        prewitty = cv2.filter2D(blurred, cv2.CV_64F, kernely)
        prewitt = np.sqrt(prewittx**2 + prewitty**2)
        return np.uint8(np.clip(prewitt, 0, 255))
    
    elif method == 'roberts':
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        padded = cv2.copyMakeBorder(blurred, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
        robertsx = cv2.filter2D(padded, cv2.CV_64F, kernelx)
        robertsy = cv2.filter2D(padded, cv2.CV_64F, kernely)
        roberts_x_sliced = robertsx[:-1, :-1]
        roberts_y_sliced = robertsy[:-1, :-1]
        roberts = np.sqrt(roberts_x_sliced**2 + roberts_y_sliced**2)
        return np.uint8(np.clip(roberts, 0, 255))
    
    elif method == 'canny':
        return cv2.Canny(blurred, 50, 150)
    
    return gray

def get_kernel(shape, ksize):
    """Membuat kernel morfologi sesuai bentuk dan ukuran."""
    if ksize % 2 == 0: ksize +=1 # Pastikan ganjil
    if ksize < 3: ksize = 3 # Ukuran minimal
    if shape == 'persegi':
        return np.ones((ksize, ksize), np.uint8)
    elif shape == 'lingkaran':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    elif shape == 'segitiga':
        kernel = np.zeros((ksize, ksize), np.uint8)
        for i in range(ksize):
            width_at_row = i + 1 # Lebar meningkat dari 1 hingga ksize (segitiga bawah)
            # Jika ingin segitiga atas: width_at_row = ksize - i
            start_col = (ksize - width_at_row) // 2
            end_col = start_col + width_at_row
            if i < ksize : # Modifikasi untuk segitiga yang lebih umum
                 kernel[i, start_col:end_col] = 1
        if np.sum(kernel) == 0: return np.ones((3,3), np.uint8) # fallback jika ksize kecil
        return kernel
    return np.ones((ksize, ksize), np.uint8)

def apply_morphology(image, morph_type, kernel_shape, kernel_size):
    """Menerapkan operasi morfologi (erosi/dilasi) pada gambar."""
    if image is None: return None
    binary_image = _ensure_binary(image) # Pastikan input adalah biner
    if binary_image is None: return None
    
    kernel = get_kernel(kernel_shape, kernel_size)
    if morph_type == 'erosi':
        return cv2.erode(binary_image, kernel, iterations=1)
    elif morph_type == 'dilasi':
        return cv2.dilate(binary_image, kernel, iterations=1)
    return binary_image

def _ensure_binary(image, threshold_val=127):
    """Helper untuk memastikan gambar adalah biner (objek putih, background hitam)."""
    if image is None: return None
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    return binary

def morphology_boundary(image):
    binary = _ensure_binary(image)
    if binary is None: return None
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return cv2.subtract(binary, eroded)

def morphology_skeleton(image):
    img_binary = _ensure_binary(image)
    if img_binary is None: return None
    skel = np.zeros(img_binary.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    temp_img = img_binary.copy()
    while True:
        eroded = cv2.erode(temp_img, kernel)
        temp_dilated = cv2.dilate(eroded, kernel)
        temp_sub = cv2.subtract(temp_img, temp_dilated)
        skel = cv2.bitwise_or(skel, temp_sub)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0:
            break
    return skel

def morphology_thickening(image):
    binary = _ensure_binary(image)
    if binary is None: return None
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    # Untuk "true" thickening, algoritma lebih kompleks. Ini adalah highlight piksel yang ditambahkan.
    return cv2.subtract(dilated, binary)

def morphology_regionfill(image_input):
    binary_obj_white = _ensure_binary(image_input) # Objek putih, bg hitam
    if binary_obj_white is None: return None

    # Buat citra terbalik: objek hitam, bg putih (untuk floodFill dari border)
    binary_obj_black_bg_white = cv2.bitwise_not(binary_obj_white)
    
    im_floodfill = binary_obj_black_bg_white.copy()
    h, w = binary_obj_black_bg_white.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill dari border (0,0) pada citra objek hitam, bg putih
    
    cv2.floodFill(im_floodfill, mask, (0,0), 0) # Isi dengan 0 (hitam)

    
    
    # Cara lain: fill dari contour
    filled_image = binary_obj_white.copy()
    contours, hierarchy = cv2.findContours(binary_obj_white, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # Cari semua kontur
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            # Jika kontur adalah hole (memiliki parent di hierarki CCOMP)
            if hierarchy[0][i][3] != -1: # Indeks ke-3 adalah parent
                cv2.drawContours(filled_image, [contour], -1, 255, thickness=cv2.FILLED)
    return filled_image


def morphology_convexhull(image):
    binary = _ensure_binary(image)
    if binary is None: return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_img = np.zeros_like(binary)
    for cnt in contours:
        hull_points = cv2.convexHull(cnt)
        cv2.drawContours(hull_img, [hull_points], 0, 255, -1)
    return hull_img

def morphology_purning(image):
    skel = morphology_skeleton(image)
    if skel is None: return None
    kernel = np.ones((3,3), np.uint8)
    return cv2.erode(skel, kernel, iterations=1)

def morphology_thinning(image):
    """Apply morphological thinning operation using Zhang-Suen algorithm."""
    try:
        # Import ximgproc from contrib module
        from cv2 import ximgproc
    except ImportError:
        app.logger.warning("cv2.ximgproc not available. Using skeleton method instead.")
        return morphology_skeleton(image)

    binary = _ensure_binary(image)
    if binary is None: 
        return None
        
    try:
        # Apply Zhang-Suen thinning
        thinned = ximgproc.thinning(
            binary, 
            thinningType=ximgproc.THINNING_ZHANGSUEN
        )
        return thinned
    except Exception as e:
        app.logger.error(f"Thinning error: {str(e)}")
        # Fallback to morphological skeleton
        return morphology_skeleton(image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Tidak ada file yang diunggah.', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('Tidak ada file yang dipilih.', 'warning')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        original_filename_unsafe = file.filename
        filename_parts = os.path.splitext(original_filename_unsafe)
        base_name = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in filename_parts[0])
        extension = filename_parts[1]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{base_name}{extension}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Pastikan folder ada
            file.save(file_path)
        except Exception as e:
            flash(f"Gagal menyimpan file unggahan: {e}", "danger")
            app.logger.error(f"File save error for {filename}: {e}")
            return redirect(url_for('index'))

        session['original_image_filename'] = filename
        session['original_image_path'] = file_path
        session['selected_method'] = request.form.get('selected_method', 'threshold')
        
        flash('File berhasil diunggah!', 'success')
        return redirect(url_for('process_image'))
    else:
        flash('Format file tidak diizinkan. Gunakan: png, jpg, jpeg, bmp, tif, tiff.', 'danger')
        return redirect(url_for('index'))

@app.route('/process', methods=['GET', 'POST'])
def process_image():
    if 'original_image_path' not in session or not os.path.exists(session['original_image_path']):
        flash('Silakan unggah gambar terlebih dahulu atau file asli tidak ditemukan.', 'warning')
        return redirect(url_for('index'))
    
    original_path = session['original_image_path']
    original_image_filename = session.get('original_image_filename', 'unknown_original.jpg')
    
    try:
        original_image_cv = cv2.imread(original_path)
        if original_image_cv is None:
            flash(f'Gagal memuat gambar dari: {original_path}. File mungkin rusak atau tidak didukung.', 'danger')
            session.pop('original_image_path', None)
            session.pop('original_image_filename', None)
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Terjadi kesalahan saat membaca gambar: {str(e)}', 'danger')
        return redirect(url_for('index'))

    selected_method = session.get('selected_method', 'threshold')
    
    original_base64 = convert_to_base64(original_path)
    if original_base64 is None:
        flash('Gagal mengkonversi gambar asli untuk ditampilkan.', 'danger')
        return redirect(url_for('index'))
        
    results = {'original': original_base64}
    processed_image_cv = None 
    method_name_key = "" 
    result_filename_suffix = ""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if selected_method == 'threshold':
        threshold_value = 127 
        if request.method == 'POST':
            threshold_value = int(request.form.get('threshold_value', 127))
        processed_image_cv = apply_threshold(original_image_cv.copy(), threshold_value)
        method_name_key = f'Threshold ({threshold_value})'
        result_filename_suffix = f"threshold_{threshold_value}"
    
    elif selected_method == 'morphology':
        morph_type = 'erosi' 
        kernel_shape = 'persegi' 
        kernel_size = 3 
        if request.method == 'POST':
            morph_type = request.form.get('morph_type', 'erosi')
            kernel_shape = request.form.get('kernel_shape', 'persegi')
            kernel_size = int(request.form.get('kernel_size', 3))
        processed_image_cv = apply_morphology(original_image_cv.copy(), morph_type, kernel_shape, kernel_size)
        method_name_key = f'{morph_type.capitalize()} ({kernel_shape.capitalize()} {kernel_size}x{kernel_size})'
        result_filename_suffix = f"{morph_type}_{kernel_shape}_{kernel_size}"

    elif selected_method == 'edge':
        edge_methods_to_run = ['sobel', 'prewitt', 'roberts', 'canny']
        for emethod in edge_methods_to_run:
            try:
                result_img = apply_edge_detection(original_image_cv.copy(), emethod)
                if result_img is not None:
                    res_filename = f"{timestamp}_{emethod}_{original_image_filename}"
                    res_path = save_image(result_img, res_filename, app.config['RESULT_FOLDER'])
                    if res_path:
                        b64_img = convert_to_base64(res_path)
                        if b64_img:
                            results[emethod.capitalize()] = b64_img
                        else:
                             flash(f'Gagal konversi base64 untuk {emethod.capitalize()}', 'warning')
                    else:
                        flash(f'Gagal menyimpan hasil untuk {emethod.capitalize()}', 'warning')
                else:
                    flash(f'Hasil {emethod.capitalize()} adalah None (tidak ada gambar).', 'warning')
            except Exception as e:
                flash(f'Error memproses deteksi tepi {emethod}: {str(e)}', 'danger')
                app.logger.error(f"Edge detection error ({emethod}): {e}")
    
    else: 
        adv_morph_map = {
            'morphology_boundary': (morphology_boundary, 'Boundary Detection', 'boundary'),
            'morphology_skeleton': (morphology_skeleton, 'Skeletonization', 'skeleton'),
            'morphology_thickening': (morphology_thickening, 'Thickening (Highlight)', 'thickening_highlight'),
            'morphology_regionfill': (morphology_regionfill, 'Region Filling', 'region_fill'),
            'morphology_convexhull': (morphology_convexhull, 'Convex Hull', 'convex_hull'),
            'morphology_purning': (morphology_purning, 'Pruning (Simple)', 'pruning_simple'),
            'morphology_thinning': (morphology_thinning, 'Thinning', 'thinning')
        }
        if selected_method in adv_morph_map:
            func, display_name, file_suffix = adv_morph_map[selected_method]
            try:
                processed_image_cv = func(original_image_cv.copy())
                method_name_key = display_name
                result_filename_suffix = file_suffix
            except Exception as e:
                flash(f'Error saat pemrosesan {display_name}: {str(e)}', 'danger')
                app.logger.error(f"Advanced Morphology error ({selected_method}): {e}")

    if processed_image_cv is not None and method_name_key and result_filename_suffix:
        result_full_filename = f"{timestamp}_{result_filename_suffix}_{original_image_filename}"
        result_path = save_image(processed_image_cv, result_full_filename, app.config['RESULT_FOLDER'])
        if result_path:
            base64_res = convert_to_base64(result_path)
            if base64_res:
                results[method_name_key] = base64_res
            else:
                flash(f'Gagal mengkonversi hasil {method_name_key} untuk ditampilkan.', 'warning')
        else:
            flash(f'Gagal menyimpan hasil untuk {method_name_key}.', 'warning')
            
    return render_template('process.html', results=results, selected_method=selected_method)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    app.logger.info("Starting Flask Application...") # Contoh penggunaan app.logger
    app.run(debug=True)