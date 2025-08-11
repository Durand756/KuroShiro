import os
import shutil
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
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
            
            # Nettoyer le dossier uploads
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    if file_time < cutoff_time:
                        os.remove(filepath)
            
            # Nettoyer le dossier processed
            for filename in os.listdir(PROCESSED_FOLDER):
                filepath = os.path.join(PROCESSED_FOLDER, filename)
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        
        except Exception as e:
            print(f"Erreur lors du nettoyage: {e}")
        
        # Attendre 30 minutes avant le prochain nettoyage
        time.sleep(1800)

# Lancer le thread de nettoyage en arrière-plan
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

def remove_background_simple(image):
    """
    Version améliorée de suppression de fond pour dessins
    Optimisée pour des résultats lisses et propres
    """
    try:
        height, width = image.shape[:2]
        
        # Méthode 1: Détection du fond par analyse des bords
        # Échantillonner les pixels des bords pour détecter la couleur de fond
        border_size = min(20, min(height, width) // 10)
        
        # Extraire les pixels des bords
        top_border = image[:border_size, :].reshape(-1, 3)
        bottom_border = image[-border_size:, :].reshape(-1, 3)
        left_border = image[:, :border_size].reshape(-1, 3)
        right_border = image[:, -border_size:].reshape(-1, 3)
        
        border_pixels = np.vstack([top_border, bottom_border, left_border, right_border])
        
        # Calculer la couleur dominante du fond (médiane pour robustesse)
        bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
        
        print(f"Couleur de fond détectée: {bg_color}")
        
        # Méthode 2: Créer un masque basé sur la distance de couleur
        # Convertir en LAB pour une meilleure perception des couleurs
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_bg_color = cv2.cvtColor(bg_color.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0]
        
        # Calculer la distance euclidienne dans l'espace LAB
        diff = lab_image.astype(np.float32) - lab_bg_color.astype(np.float32)
        distance = np.sqrt(np.sum(diff**2, axis=2))
        
        # Seuillage adaptatif basé sur l'écart-type de la distance
        threshold = np.mean(distance) + 0.8 * np.std(distance)
        mask = (distance > threshold).astype(np.uint8) * 255
        
        # Méthode 3: Améliorer le masque avec des techniques morphologiques
        # Fermeture pour boucher les petits trous
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Ouverture pour supprimer le bruit
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # Dilatation légère pour inclure les bords fins
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        # Méthode 4: Lissage du masque pour des bords plus doux
        # Flou gaussien pour adoucir les bords
        mask_smooth = cv2.GaussianBlur(mask, (5, 5), 2)
        
        # Méthode 5: Détection des contours principaux pour affiner
        contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Garder seulement les gros contours (objets principaux)
        min_area = (width * height) * 0.01  # Au moins 1% de l'image
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if large_contours:
            # Créer un masque propre avec les contours principaux
            mask_clean = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask_clean, large_contours, 255)
            
            # Lissage final
            mask_final = cv2.GaussianBlur(mask_clean, (3, 3), 1)
        else:
            # Si pas de gros contours, utiliser le masque lissé
            mask_final = mask_smooth
        
        # Créer l'image résultat avec fond transparent
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = mask_final  # Canal alpha
        
        return result, mask_final
        
    except Exception as e:
        print(f"Erreur dans remove_background_simple: {e}")
        import traceback
        traceback.print_exc()
        
        # En cas d'erreur, utiliser une méthode de fallback simple
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Seuillage simple: supposer que le fond est plus clair
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Si l'objet est sombre sur fond clair, inverser le masque
        if np.mean(gray[mask == 255]) < 128:
            mask = cv2.bitwise_not(mask)
        
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = mask
        
        return result, mask

def process_image_to_manga(image_path, output_path):
    """
    Version améliorée pour des résultats manga lisses et propres
    """
    try:
        # 1. Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Impossible de charger l'image")
        
        print(f"Image chargée: {img.shape}")
        
        # 2. Prétraitement pour améliorer la qualité
        # Réduction du bruit léger sans flouter les détails
        img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        # 3. Supprimer le fond (version améliorée)
        img_no_bg, alpha_mask = remove_background_simple(img_denoised)
        
        # 4. Convertir en niveaux de gris de manière optimale
        if len(img_no_bg.shape) == 4:  # RGBA
            gray = cv2.cvtColor(img_no_bg[:,:,:3], cv2.COLOR_BGR2GRAY)
        else:  # RGB
            gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
            alpha_mask = np.ones(gray.shape, dtype=np.uint8) * 255
        
        print(f"Conversion gris: {gray.shape}")
        
        # 5. Amélioration du contraste adaptative
        # CLAHE avec paramètres optimisés pour les dessins
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 6. Préparation pour un seuillage optimal
        # Flou léger pour réduire le bruit de numérisation
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 7. Seuillage adaptatif optimisé pour dessins
        # Utiliser plusieurs méthodes et les combiner
        
        # Méthode 1: Seuillage adaptatif Gaussian
        binary1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 4)
        
        # Méthode 2: Seuillage adaptatif Mean
        binary2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 15, 6)
        
        # Méthode 3: Seuillage Otsu pour les zones uniformes
        _, binary3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combiner les méthodes (intersection pour plus de précision)
        binary_combined = cv2.bitwise_and(binary1, binary2)
        
        # Utiliser Otsu pour les zones où les autres méthodes donnent du blanc
        mask_otsu = binary_combined == 255
        binary_combined[mask_otsu] = binary3[mask_otsu]
        
        print(f"Seuillage combiné effectué: {binary_combined.shape}")
        
        # 8. Nettoyage morphologique sophistiqué
        # Éléments structurants plus adaptés aux dessins
        kernel_line_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        kernel_line_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Fermeture pour connecter les lignes brisées
        binary_clean = cv2.morphologyEx(binary_combined, cv2.MORPH_CLOSE, kernel_small)
        
        # Fermetures directionnelles pour les lignes fines
        temp1 = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel_line_horizontal)
        temp2 = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel_line_vertical)
        binary_clean = cv2.bitwise_or(temp1, temp2)
        
        # Ouverture pour supprimer le bruit ponctuel
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel_small)
        
        # 9. Amélioration des contours
        # Détection de contours avec paramètres optimisés
        edges = cv2.Canny(enhanced, 30, 100, apertureSize=3, L2gradient=True)
        
        # Dilatation légère des contours pour les épaissir
        edges_thick = cv2.dilate(edges, kernel_small, iterations=1)
        
        # Combiner seuillage et contours
        combined = cv2.bitwise_or(binary_clean, edges_thick)
        
        # 10. Application du masque alpha de manière lisse
        result = np.ones_like(combined) * 255  # Fond blanc
        
        # Créer une transition douce avec le masque alpha
        alpha_normalized = alpha_mask.astype(np.float32) / 255.0
        
        # Appliquer le masque avec transition douce
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if alpha_normalized[i, j] > 0.1:  # Seuil pour éviter le bruit
                    # Mélange proportionnel basé sur l'alpha
                    result[i, j] = int(255 * (1 - alpha_normalized[i, j]) + 
                                     combined[i, j] * alpha_normalized[i, j])
        
        # 11. Ajout d'effets manga sophistiqués
        result_with_effects = add_manga_effects_improved(result, alpha_mask, enhanced)
        
        print(f"Effets manga appliqués: {result_with_effects.shape}")
        
        # 12. Post-traitement final pour la netteté
        # Filtre de netteté adaptatif
        kernel_sharpen = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]]) / 9.0 * 1.2
        
        result_sharp = cv2.filter2D(result_with_effects.astype(np.float32), -1, kernel_sharpen)
        result_sharp = np.clip(result_sharp, 0, 255).astype(np.uint8)
        
        # Mélange subtil pour éviter l'over-sharpening
        final_result = cv2.addWeighted(result_with_effects, 0.8, result_sharp, 0.2, 0)
        
        # 13. Lissage final pour supprimer les artefacts
        final_result = cv2.medianBlur(final_result, 3)
        
        # 14. Sauvegarder le résultat
        cv2.imwrite(output_path, final_result)
        print(f"Image sauvegardée avec succès: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_manga_effects_improved(image, alpha_channel, original_gray):
    """
    Version améliorée des effets manga pour des résultats lisses et professionnels
    """
    try:
        result = image.copy()
        height, width = image.shape
        
        # 1. Analyser les zones pour un placement intelligent des trames
        # Créer une carte des niveaux de gris
        gray_levels = cv2.GaussianBlur(original_gray, (5, 5), 0)
        
        # 2. Trames de points sophistiquées avec variation de densité
        dot_patterns = [
            {'size': 2, 'spacing': 8, 'gray_min': 140, 'gray_max': 180, 'density': 0.7},
            {'size': 3, 'spacing': 12, 'gray_min': 100, 'gray_max': 140, 'density': 0.5},
            {'size': 4, 'spacing': 16, 'gray_min': 60, 'gray_max': 100, 'density': 0.3}
        ]
        
        for pattern in dot_patterns:
            dot_size = pattern['size']
            spacing = pattern['spacing']
            gray_min = pattern['gray_min']
            gray_max = pattern['gray_max']
            density = pattern['density']
            
            # Placement des points avec offset pour éviter la régularité
            offset_x = spacing // 4
            offset_y = spacing // 4
            
            for y in range(offset_y, height - spacing//2, spacing):
                for x in range(offset_x, width - spacing//2, spacing):
                    if x < width and y < height:
                        pixel_val = image[y, x]
                        alpha_val = alpha_channel[y, x] if alpha_channel is not None else 255
                        original_val = gray_levels[y, x]
                        
                        # Vérifier si on est dans la bonne zone de gris
                        if (gray_min <= original_val <= gray_max and 
                            alpha_val > 128 and pixel_val > 200):
                            
                            # Probabilité variable selon l'intensité
                            probability = density * ((gray_max - original_val) / (gray_max - gray_min))
                            
                            if np.random.random() < probability:
                                # Ajouter de la variation dans la position
                                jitter_x = int(np.random.normal(0, spacing//8))
                                jitter_y = int(np.random.normal(0, spacing//8))
                                center_x = x + jitter_x
                                center_y = y + jitter_y
                                
                                # Dessiner un point avec dégradé pour plus de réalisme
                                for dy in range(-dot_size, dot_size + 1):
                                    for dx in range(-dot_size, dot_size + 1):
                                        ny, nx = center_y + dy, center_x + dx
                                        if 0 <= ny < height and 0 <= nx < width:
                                            distance = np.sqrt(dx*dx + dy*dy)
                                            if distance <= dot_size:
                                                # Gradient du point (plus foncé au centre)
                                                intensity = max(0, 1 - distance / dot_size)
                                                current_val = result[ny, nx]
                                                new_val = int(current_val * (1 - intensity * 0.8))
                                                result[ny, nx] = max(0, new_val)
        
        # 3. Hachures directionnelles sophistiquées
        hatch_patterns = [
            {'angle': 45, 'spacing': 6, 'thickness': 1, 'gray_min': 40, 'gray_max': 80},
            {'angle': -45, 'spacing': 8, 'thickness': 1, 'gray_min': 20, 'gray_max': 60},
            {'angle': 90, 'spacing': 10, 'thickness': 2, 'gray_min': 0, 'gray_max': 40}
        ]
        
        for pattern in hatch_patterns:
            angle = pattern['angle']
            spacing = pattern['spacing']
            thickness = pattern['thickness']
            gray_min = pattern['gray_min']
            gray_max = pattern['gray_max']
            
            # Créer une matrice de rotation
            angle_rad = np.radians(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            for y in range(height):
                for x in range(width):
                    original_val = gray_levels[y, x]
                    alpha_val = alpha_channel[y, x] if alpha_channel is not None else 255
                    
                    if (gray_min <= original_val <= gray_max and alpha_val > 128):
                        # Coordonnées rotées
                        rot_x = cos_a * x - sin_a * y
                        
                        # Vérifier si on est sur une ligne de hachure
                        if int(rot_x) % spacing < thickness:
                            # Intensité variable selon le niveau de gris
                            intensity = (gray_max - original_val) / (gray_max - gray_min)
                            if np.random.random() < intensity:
                                result[y, x] = max(0, result[y, x] - 100)
        
        # 4. Amélioration des contours existants
        # Détecter les bordures des zones noires
        kernel_edge = np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]])
        
        edges = cv2.filter2D((255 - result).astype(np.float32), -1, kernel_edge)
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        
        # Renforcer légèrement les contours
        mask_edges = edges > 30
        result[mask_edges] = np.maximum(result[mask_edges] - 50, 0)
        
        # 5. Lissage final pour éliminer les artefacts
        result = cv2.medianBlur(result, 3)
        
        return result
        
    except Exception as e:
        print(f"Erreur lors de l'ajout des effets manga: {e}")
        import traceback
        traceback.print_exc()
        return image((2,2), np.uint8), iterations=1)
        result = cv2.bitwise_and(result, cv2.bitwise_not(enhanced_edges))
        
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
    """Gère l'upload et le traitement de l'image"""
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
            output_filename = 'manga_' + filename
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            
            # Traiter l'image
            success = process_image_to_manga(upload_path, output_path)
            
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
    print("🎨 Serveur Manga Converter démarré!")
    print("📁 Dossiers créés: uploads/, processed/")
    print("🧹 Nettoyage automatique des fichiers anciens activé")
    print("🌐 Accès: http://localhost:5000")
    print("⚠️  Version sans rembg (plus stable pour le déploiement)")
    app.run(debug=True, host='0.0.0.0', port=5000)
