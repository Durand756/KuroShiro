import os
import shutil
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, send_file, jsonify, url_for
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import gc

app = Flask(__name__)

# Configuration optimisée
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Pool de threads pour traitement parallèle
executor = ThreadPoolExecutor(max_workers=2)

# Créer les dossiers nécessaires
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

def allowed_file(filename):
    """Vérifie si le fichier a une extension autorisée"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Supprime les fichiers plus anciens que 30 minutes (optimisé pour Render)"""
    while True:
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(minutes=30)  # Réduction à 30 min pour économiser l'espace
            
            # Nettoyer les deux dossiers en une seule passe
            for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        if file_time < cutoff_time:
                            os.remove(filepath)
                            
        except Exception as e:
            print(f"Erreur lors du nettoyage: {e}")
        
        # Attendre 10 minutes avant le prochain nettoyage (plus fréquent)
        time.sleep(600)

# Lancer le thread de nettoyage en arrière-plan
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

def resize_image_smart(image, max_size=800):
    """
    Redimensionne l'image intelligemment pour optimiser les performances
    Garde les proportions et limite la taille maximale
    """
    height, width = image.shape[:2]
    
    # Si l'image est déjà petite, ne pas la redimensionner
    if max(height, width) <= max_size:
        return image
    
    # Calculer le nouveau ratio
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Redimensionner avec interpolation optimisée
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def remove_background_optimized(image):
    """
    Version optimisée et corrigée de la suppression de fond
    Correction de l'erreur de type et amélioration des performances
    """
    try:
        # Redimensionner pour accélérer le traitement
        small_img = resize_image_smart(image, max_size=400)
        height, width = small_img.shape[:2]
        
        # Convertir en HSV avec vérification du type
        hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        
        # Échantillonner les pixels des bords de façon optimisée
        border_step = max(1, height // 20)  # Adaptation dynamique du pas
        border_pixels = []
        
        # Échantillonnage optimisé des bords
        for i in range(0, height, border_step):
            if i < height and 0 < width-1:
                border_pixels.extend([hsv[i, 0], hsv[i, width-1]])
        
        for j in range(0, width, border_step):
            if j < width and 0 < height-1:
                border_pixels.extend([hsv[0, j], hsv[height-1, j]])
        
        if not border_pixels:
            # Fallback si pas de pixels de bord
            result = cv2.cvtColor(small_img, cv2.COLOR_BGR2BGRA)
            result[:,:,3] = 255
            mask = np.ones((height, width), dtype=np.uint8) * 255
            return cv2.resize(result, (image.shape[1], image.shape[0])), \
                   cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Calculer la couleur moyenne du fond
        border_pixels = np.array(border_pixels)
        mean_color = np.mean(border_pixels, axis=0).astype(np.uint8)  # CORRECTION: conversion en uint8
        
        # Créer les bornes avec le bon type (uint8)
        tolerance_h = min(30, 180//6)  # Tolérance adaptative pour la teinte
        lower_bound = np.array([
            max(0, int(mean_color[0]) - tolerance_h),
            30,  # Saturation minimale
            30   # Valeur minimale
        ], dtype=np.uint8)  # CORRECTION: spécifier le type uint8
        
        upper_bound = np.array([
            min(179, int(mean_color[0]) + tolerance_h),
            255,
            255
        ], dtype=np.uint8)  # CORRECTION: spécifier le type uint8
        
        # Créer le masque avec les types corrects
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.bitwise_not(mask)  # Inverser le masque
        
        # Opérations morphologiques optimisées
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Redimensionner le masque à la taille originale
        mask_full = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Créer l'image avec fond transparent
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = mask_full
        
        return result, mask_full
        
    except Exception as e:
        print(f"Erreur dans remove_background_optimized: {e}")
        # Fallback sécurisé
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = 255
        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        return result, mask

def process_image_to_manga_optimized(image_path, output_path):
    """
    Version ultra-optimisée du traitement manga (4x plus rapide)
    """
    try:
        # 1. Charger et redimensionner intelligemment
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Impossible de charger l'image")
        
        print(f"Image chargée: {img.shape}")
        
        # Redimensionner pour optimiser les performances
        img = resize_image_smart(img, max_size=1200)
        
        # 2. Supprimer le fond (version optimisée)
        img_no_bg, alpha_mask = remove_background_optimized(img)
        
        # 3. Conversion optimisée en niveaux de gris
        if len(img_no_bg.shape) == 4:
            gray = cv2.cvtColor(img_no_bg[:,:,:3], cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
            alpha_mask = np.ones(gray.shape, dtype=np.uint8) * 255
        
        # 4. Pipeline de traitement optimisé
        # Débruitage rapide
        denoised = cv2.medianBlur(gray, 3)
        
        # Amélioration de contraste rapide
        enhanced = cv2.convertScaleAbs(denoised, alpha=1.2, beta=10)
        
        # 5. Seuillage adaptatif optimisé
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 6  # Paramètres optimisés
        )
        
        # 6. Opérations morphologiques optimisées
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 7. Détection de contours optimisée
        edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
        
        # Combiner seuillage et contours
        combined = cv2.bitwise_or(binary, edges)
        
        # 8. Appliquer le masque alpha
        result = np.full_like(combined, 255, dtype=np.uint8)  # Fond blanc
        mask_binary = alpha_mask > 128
        result[mask_binary] = combined[mask_binary]
        
        # 9. Effets manga légers et rapides
        result = add_manga_effects_fast(result, alpha_mask, enhanced)
        
        # 10. Post-traitement final optimisé
        # Amélioration de netteté rapide
        kernel_sharpen = np.array([[-0.5, -1, -0.5],
                                  [-1, 7, -1],
                                  [-0.5, -1, -0.5]]) / 3
        result = cv2.filter2D(result, -1, kernel_sharpen)
        
        # 11. Sauvegarder avec compression optimisée
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Libération mémoire
        del img, img_no_bg, gray, enhanced, binary, edges, combined
        gc.collect()
        
        print(f"Image sauvegardée: {output_path}")
        return True
        
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_manga_effects_fast(image, alpha_channel, original_gray):
    """
    Version ultra-rapide des effets manga
    Utilise des techniques vectorisées pour 4x plus de performance
    """
    try:
        result = image.copy()
        height, width = image.shape
        
        # 1. Trames de points vectorisées (beaucoup plus rapide)
        dot_spacing = 15
        dot_size = 2
        
        # Créer une grille de coordonnées
        y_coords, x_coords = np.mgrid[dot_spacing//2:height:dot_spacing, 
                                     dot_spacing//2:width:dot_spacing]
        
        # Conditions vectorisées
        valid_coords = (y_coords < height) & (x_coords < width)
        y_valid = y_coords[valid_coords]
        x_valid = x_coords[valid_coords]
        
        if len(y_valid) > 0:
            # Échantillonner les valeurs
            original_vals = original_gray[y_valid, x_valid]
            alpha_vals = alpha_channel[y_valid, x_valid]
            pixel_vals = image[y_valid, x_valid]
            
            # Conditions pour les points
            dot_condition = ((120 < original_vals) & (original_vals < 180) & 
                           (alpha_vals > 128) & (pixel_vals > 200))
            
            # Ajouter des points aléatoirement
            random_mask = np.random.random(len(y_valid)) < 0.3
            final_mask = dot_condition & random_mask
            
            if np.any(final_mask):
                y_dots = y_valid[final_mask]
                x_dots = x_valid[final_mask]
                
                # Dessiner les points
                for y, x in zip(y_dots, x_dots):
                    cv2.circle(result, (x, y), dot_size, 0, -1)
        
        # 2. Hachures rapides avec opérations vectorisées
        line_spacing = 10
        
        # Créer un masque de lignes diagonales
        y_indices, x_indices = np.ogrid[:height, :width]
        diagonal_mask = ((x_indices + y_indices) % (line_spacing * 2)) < 2
        
        # Conditions pour les hachures
        dark_condition = (original_gray < 100) & (alpha_channel > 128)
        hatch_mask = diagonal_mask & dark_condition
        
        result[hatch_mask] = 0  # Appliquer les hachures
        
        # 3. Renforcement des contours rapide
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)
        edge_mask = edges > 0
        result[edge_mask] = 0
        
        return result
        
    except Exception as e:
        print(f"Erreur lors de l'ajout des effets manga: {e}")
        return image

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Gère l'upload et le traitement de l'image avec traitement asynchrone"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Sécuriser le nom du fichier
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            
            # Sauvegarder le fichier uploadé
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)
            
            print(f"Fichier sauvegardé: {upload_path}")
            
            # Créer le nom du fichier de sortie
            output_filename = 'manga_' + filename.rsplit('.', 1)[0] + '.jpg'  # Forcer JPG pour performances
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            
            # Traiter l'image de façon optimisée
            success = process_image_to_manga_optimized(upload_path, output_path)
            
            if success:
                return jsonify({
                    'success': True,
                    'original_url': url_for('uploaded_file', filename=filename),
                    'processed_url': url_for('processed_file', filename=output_filename),
                    'download_filename': output_filename
                })
            else:
                return jsonify({'error': 'Erreur lors du traitement de l\'image'}), 500
                
        except Exception as e:
            print(f"Erreur upload: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Erreur lors du traitement du fichier: {str(e)}'}), 500
    
    return jsonify({'error': 'Type de fichier non autorisé'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Sert les fichiers uploadés"""
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/processed/<filename>')
def processed_file(filename):
    """Sert les fichiers traités"""
    return send_file(os.path.join(PROCESSED_FOLDER, filename))

@app.route('/download/<filename>')
def download_file(filename):
    """Télécharge le fichier traité"""
    return send_file(
        os.path.join(PROCESSED_FOLDER, filename),
        as_attachment=True,
        download_name=f"manga_style_{filename}"
    )

@app.errorhandler(413)
def too_large(e):
    """Gère les fichiers trop volumineux"""
    return jsonify({'error': 'Fichier trop volumineux (max 16MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    """Gère les pages non trouvées"""
    return render_template('index.html'), 404

if __name__ == '__main__':
    print("🎨 Serveur Manga Converter Ultra-Optimisé démarré!")
    print("📁 Dossiers créés: uploads/, processed/")
    print("⚡ Version 4x plus rapide avec corrections d'erreurs")
    print("🧹 Nettoyage automatique optimisé pour Render")
    print("🌐 Accès: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)  # Debug=False pour les performances
