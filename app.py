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
    Alternative simple à rembg pour supprimer le fond
    Utilise la détection de contours et assume que le fond est la couleur dominante
    """
    try:
        # Convertir en HSV pour une meilleure détection de couleur
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Créer un masque basé sur la couleur dominante (supposée être le fond)
        # Calculer la couleur moyenne des bords (probablement le fond)
        height, width = image.shape[:2]
        border_pixels = []
        
        # Échantillonner les pixels des bords
        for i in range(0, height, 10):
            border_pixels.append(hsv[i, 0])  # Bord gauche
            border_pixels.append(hsv[i, width-1])  # Bord droit
        
        for j in range(0, width, 10):
            border_pixels.append(hsv[0, j])  # Bord haut
            border_pixels.append(hsv[height-1, j])  # Bord bas
        
        # Calculer la couleur moyenne du fond
        border_pixels = np.array(border_pixels)
        mean_color = np.mean(border_pixels, axis=0)
        
        # Créer un masque pour les pixels similaires au fond
        lower_bound = np.array([max(0, mean_color[0]-20), 50, 50])
        upper_bound = np.array([min(179, mean_color[0]+20), 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Inverser le masque (nous voulons garder l'objet, pas le fond)
        mask = cv2.bitwise_not(mask)
        
        # Appliquer des opérations morphologiques pour nettoyer le masque
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Créer l'image avec fond transparent
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = mask  # Canal alpha
        
        return result, mask
        
    except Exception as e:
        print(f"Erreur dans remove_background_simple: {e}")
        # En cas d'erreur, retourner l'image originale avec un masque complet
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = 255  # Alpha complet
        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        return result, mask

def process_image_to_manga(image_path, output_path):
    """
    Traite une image pour la convertir en style manga/BD
    Version sans rembg pour plus de stabilité
    """
    try:
        # 1. Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Impossible de charger l'image")
        
        print(f"Image chargée: {img.shape}")
        
        # 2. Supprimer le fond (version simplifiée)
        img_no_bg, alpha_mask = remove_background_simple(img)
        
        # 3. Convertir en niveaux de gris
        if len(img_no_bg.shape) == 4:  # RGBA
            gray = cv2.cvtColor(img_no_bg[:,:,:3], cv2.COLOR_BGR2GRAY)
        else:  # RGB
            gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
            alpha_mask = np.ones(gray.shape, dtype=np.uint8) * 255
        
        print(f"Conversion gris: {gray.shape}")
        
        # 4. Améliorer le contraste avec CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 5. Réduction du bruit avant seuillage
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 6. Seuillage adaptatif pour obtenir du noir et blanc pur
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 15, 8)
        
        print(f"Seuillage effectué: {binary.shape}")
        
        # 7. Opérations morphologiques pour nettoyer
        kernel_small = np.ones((2,2), np.uint8)
        kernel_medium = np.ones((3,3), np.uint8)
        
        # Fermeture pour connecter les lignes brisées
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        # Ouverture pour supprimer le bruit
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        # Dilatation légère pour renforcer les traits
        binary = cv2.dilate(binary, kernel_small, iterations=1)
        
        # 8. Amélioration spécifique pour les dessins
        # Détecter et renforcer les contours
        edges = cv2.Canny(denoised, 50, 150, apertureSize=3, L2gradient=True)
        
        # Combiner le seuillage et les contours
        combined = cv2.bitwise_or(binary, edges)
        
        # 9. Appliquer le masque alpha pour préserver la forme
        result = np.ones_like(combined) * 255  # Fond blanc
        
        # Appliquer le masque : garder seulement les zones d'intérêt
        mask_binary = alpha_mask > 128
        result[mask_binary] = combined[mask_binary]
        
        # 10. Ajouter des trames manga (version améliorée)
        result_with_effects = add_manga_effects(result, alpha_mask, enhanced)
        
        print(f"Effets manga appliqués: {result_with_effects.shape}")
        
        # 11. Post-traitement final
        # Légère amélioration de la netteté
        kernel_sharpen = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
        result_sharp = cv2.filter2D(result_with_effects, -1, kernel_sharpen)
        
        # Mélanger avec l'original pour éviter un effet trop fort
        final_result = cv2.addWeighted(result_with_effects, 0.7, result_sharp, 0.3, 0)
        
        # 12. Sauvegarder le résultat
        cv2.imwrite(output_path, final_result)
        print(f"Image sauvegardée: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_manga_effects(image, alpha_channel, original_gray):
    """
    Ajoute des effets de style manga : trames, pointillés, hachures
    """
    try:
        result = image.copy()
        height, width = image.shape
        
        # 1. Ajouter des trames de points dans les zones grises moyennes
        dot_size = 3
        spacing = 12
        
        for y in range(spacing//2, height-spacing//2, spacing):
            for x in range(spacing//2, width-spacing//2, spacing):
                if x < width and y < height:
                    # Vérifier les valeurs des pixels
                    pixel_val = image[y, x]
                    alpha_val = alpha_channel[y, x] if alpha_channel is not None else 255
                    original_val = original_gray[y, x]
                    
                    # Ajouter des points dans les zones de gris moyen
                    if (120 < original_val < 180 and alpha_val > 128 and pixel_val > 200):
                        # Probabilité basée sur l'intensité du gris
                        probability = (180 - original_val) / 60.0
                        if np.random.random() < probability:
                            # Dessiner un petit point
                            for dy in range(-dot_size//2, dot_size//2+1):
                                for dx in range(-dot_size//2, dot_size//2+1):
                                    if (dx*dx + dy*dy) <= (dot_size//2)**2:
                                        ny, nx = y + dy, x + dx
                                        if 0 <= ny < height and 0 <= nx < width:
                                            result[ny, nx] = 0  # Noir
        
        # 2. Ajouter des hachures dans les zones très sombres
        line_spacing = 8
        for y in range(0, height, line_spacing):
            for x in range(width):
                if y < height and x < width:
                    original_val = original_gray[y, x]
                    alpha_val = alpha_channel[y, x] if alpha_channel is not None else 255
                    
                    # Hachures dans les zones sombres
                    if original_val < 100 and alpha_val > 128:
                        if (x + y) % 16 < 2:  # Lignes diagonales
                            result[y, x] = 0
        
        # 3. Améliorer les contours existants
        # Détecter les bordures des objets noirs
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)
        
        # Renforcer les contours
        enhanced_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
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
