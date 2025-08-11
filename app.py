import os
import shutil
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from flask import Flask, request, render_template, send_file, jsonify, url_for
import threading
import time

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Créer les dossiers nécessaires
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

def allowed_file(filename):
    """Vérifie si le fichier a une extension autorisée"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Supprime les fichiers plus anciens que 1 heure"""
    while True:
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=1)
            
            # Nettoyer les dossiers
            for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        if file_time < cutoff_time:
                            os.remove(filepath)
                            
        except Exception as e:
            print(f"Erreur lors du nettoyage: {e}")
        
        # Attendre 30 minutes
        time.sleep(1800)

# Thread de nettoyage
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

def process_to_manga_style(image_path, output_path):
    """
    Traitement simple et efficace pour style manga
    Version ultra-stable sans fonctions complexes
    """
    try:
        print(f"🎨 Début du traitement: {image_path}")
        
        # 1. Charger l'image avec OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Impossible de charger l'image")
        
        height, width = img.shape[:2]
        print(f"📐 Dimensions: {width}x{height}")
        
        # 2. Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Suppression du fond simple et efficace
        # Détecter la couleur dominante des bords (fond)
        border_size = min(10, min(height, width) // 20)
        
        # Échantillons des bords
        top_border = img[:border_size, :].reshape(-1, 3)
        bottom_border = img[-border_size:, :].reshape(-1, 3)
        left_border = img[:, :border_size].reshape(-1, 3)
        right_border = img[:, -border_size:].reshape(-1, 3)
        
        # Couleur moyenne du fond
        all_borders = np.vstack([top_border, bottom_border, left_border, right_border])
        bg_color = np.median(all_borders, axis=0)
        
        print(f"🎨 Couleur fond détectée: {bg_color}")
        
        # Créer un masque basé sur la similarité de couleur
        diff = np.linalg.norm(img - bg_color, axis=2)
        threshold = np.mean(diff) + np.std(diff) * 0.5
        mask = (diff > threshold).astype(np.uint8) * 255
        
        # Nettoyage du masque
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Lissage du masque
        mask = cv2.GaussianBlur(mask, (3, 3), 1)
        
        # 4. Amélioration du contraste
        # CLAHE pour améliorer les détails
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 5. Réduction du bruit
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 6. Seuillage adaptatif pour noir et blanc pur
        # Combiner plusieurs méthodes
        binary1 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        binary2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 15, 4)
        
        # Intersection des deux pour plus de précision
        binary = cv2.bitwise_and(binary1, binary2)
        
        # 7. Nettoyage morphologique
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Fermeture pour connecter les traits
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        # Ouverture pour supprimer le bruit
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # 8. Renforcement des contours
        edges = cv2.Canny(denoised, 50, 150)
        edges = cv2.dilate(edges, kernel_small, iterations=1)
        
        # Combiner binary et edges
        combined = cv2.bitwise_or(binary, edges)
        
        # 9. Application du masque
        result = np.ones_like(combined) * 255  # Fond blanc
        
        # Appliquer le masque de façon simple
        object_area = mask > 128
        result[object_area] = combined[object_area]
        
        # 10. Ajout d'effets manga simples
        result = add_simple_manga_effects(result, enhanced, object_area)
        
        # 11. Amélioration finale de la netteté
        kernel_sharpen = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
        
        sharpened = cv2.filter2D(result.astype(np.float32), -1, kernel_sharpen)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Mélange subtil
        final_result = cv2.addWeighted(result, 0.7, sharpened, 0.3, 0)
        
        # 12. Lissage final très léger
        final_result = cv2.medianBlur(final_result, 3)
        
        # 13. Sauvegarder
        success = cv2.imwrite(output_path, final_result)
        if not success:
            raise Exception("Impossible de sauvegarder l'image")
        
        print(f"✅ Image sauvegardée: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur de traitement: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_simple_manga_effects(image, original_gray, mask):
    """
    Ajout d'effets manga simples et stables
    """
    try:
        result = image.copy()
        height, width = image.shape
        
        # 1. Trames de points simples
        dot_size = 2
        spacing = 12
        
        # Grille de points avec léger décalage
        for y in range(spacing//2, height-spacing//2, spacing):
            for x in range(spacing//2, width-spacing//2, spacing):
                if y < height and x < width and mask[y, x]:
                    gray_val = original_gray[y, x]
                    pixel_val = image[y, x]
                    
                    # Points dans les zones de gris moyen
                    if 120 < gray_val < 170 and pixel_val > 200:
                        probability = (170 - gray_val) / 50.0
                        
                        if np.random.random() < probability * 0.6:
                            # Dessiner un petit point
                            for dy in range(-dot_size//2, dot_size//2+1):
                                for dx in range(-dot_size//2, dot_size//2+1):
                                    ny, nx = y + dy, x + dx
                                    if (0 <= ny < height and 0 <= nx < width and 
                                        dx*dx + dy*dy <= dot_size*dot_size//4):
                                        result[ny, nx] = 0
        
        # 2. Hachures simples dans les zones sombres
        for y in range(0, height, 8):
            for x in range(width):
                if y < height and x < width and mask[y, x]:
                    gray_val = original_gray[y, x]
                    
                    if gray_val < 100:
                        # Lignes diagonales simples
                        if (x + y) % 12 < 2:
                            result[y, x] = max(0, result[y, x] - 80)
        
        # 3. Renforcement léger des contours
        kernel = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])
        
        edges = cv2.filter2D(result.astype(np.float32), -1, kernel)
        edges = np.clip(np.abs(edges), 0, 255).astype(np.uint8)
        
        # Appliquer seulement sur les contours forts
        strong_edges = edges > 30
        result[strong_edges] = np.maximum(result[strong_edges] - 30, 0)
        
        return result
        
    except Exception as e:
        print(f"Erreur effets manga: {e}")
        return image

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload et traitement"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Nom sécurisé
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            
            # Sauvegarder
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)
            
            # Traiter
            output_filename = 'manga_' + filename
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            
            success = process_to_manga_style(upload_path, output_path)
            
            if success:
                return jsonify({
                    'success': True,
                    'original_url': url_for('uploaded_file', filename=filename),
                    'processed_url': url_for('processed_file', filename=output_filename),
                    'download_filename': output_filename
                })
            else:
                return jsonify({'error': 'Erreur lors du traitement'}), 500
                
        except Exception as e:
            print(f"Erreur upload: {e}")
            return jsonify({'error': f'Erreur: {str(e)}'}), 500
    
    return jsonify({'error': 'Type de fichier non autorisé'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename))

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(PROCESSED_FOLDER, filename),
        as_attachment=True,
        download_name=f"manga_style_{filename}"
    )

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Fichier trop volumineux (max 16MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

if __name__ == '__main__':
    print("🎨 Manga Converter - Version Simple et Stable")
    print("=" * 50)
    print("📁 Dossiers prêts: uploads/, processed/")
    print("🧹 Nettoyage automatique activé")
    print("🌐 Serveur: http://localhost:5000")
    print("✨ Algorithme optimisé pour stabilité et qualité")
    app.run(debug=True, host='0.0.0.0', port=5000)
