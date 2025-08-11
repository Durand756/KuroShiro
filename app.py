import os
import shutil
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove
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

def process_image_to_manga(image_path, output_path):
    """
    Traite une image pour la convertir en style manga/BD
    """
    try:
        # 1. Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Impossible de charger l'image")
        
        # 2. Supprimer le fond avec rembg
        with open(image_path, 'rb') as input_file:
            input_data = input_file.read()
            output_data = remove(input_data)
            
        # Sauvegarder temporairement l'image sans fond
        temp_no_bg_path = image_path.replace('.', '_no_bg.')
        with open(temp_no_bg_path, 'wb') as output_file:
            output_file.write(output_data)
        
        # 3. Recharger l'image sans fond
        img_no_bg = cv2.imread(temp_no_bg_path, cv2.IMREAD_UNCHANGED)
        
        # 4. Convertir en niveaux de gris
        if len(img_no_bg.shape) == 4:  # RGBA
            gray = cv2.cvtColor(img_no_bg[:,:,:3], cv2.COLOR_BGR2GRAY)
            alpha = img_no_bg[:,:,3]
        else:  # RGB
            gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
            alpha = np.ones(gray.shape, dtype=np.uint8) * 255
        
        # 5. Améliorer le contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 6. Seuillage adaptatif pour obtenir du noir et blanc pur
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 7. Débruitage
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 8. Appliquer le masque alpha pour préserver la transparence
        mask = alpha > 128
        result = np.ones_like(binary) * 255  # Fond blanc
        result[mask] = binary[mask]
        
        # 9. Ajouter des trames manga (optionnel)
        result_with_halftone = add_manga_halftone(result, alpha)
        
        # 10. Sauvegarder le résultat
        cv2.imwrite(output_path, result_with_halftone)
        
        # Nettoyer le fichier temporaire
        if os.path.exists(temp_no_bg_path):
            os.remove(temp_no_bg_path)
            
        return True
        
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        return False

def add_manga_halftone(image, alpha_channel):
    """
    Ajoute des trames de style manga aux zones grises
    """
    try:
        # Convertir en PIL pour un traitement plus facile
        pil_img = Image.fromarray(image)
        
        # Créer un motif de trames
        width, height = pil_img.size
        halftone = Image.new('L', (width, height), 255)
        
        # Créer des points de trame
        dot_size = 4
        spacing = 8
        
        for y in range(0, height, spacing):
            for x in range(0, width, spacing):
                # Vérifier la valeur du pixel original
                if x < width and y < height:
                    pixel_val = image[y, x]
                    alpha_val = alpha_channel[y, x]
                    
                    # Ajouter des points dans les zones grises moyennes
                    if 100 < pixel_val < 200 and alpha_val > 128:
                        # Dessiner un petit cercle
                        for dy in range(-dot_size//2, dot_size//2):
                            for dx in range(-dot_size//2, dot_size//2):
                                if (dx*dx + dy*dy) <= (dot_size//2)**2:
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < height and 0 <= nx < width:
                                        if np.random.random() > 0.3:  # 70% de chance de dessiner le point
                                            image[ny, nx] = 0  # Noir
        
        return image
        
    except Exception as e:
        print(f"Erreur lors de l'ajout des trames: {e}")
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
            return jsonify({'error': 'Erreur lors du traitement du fichier'}), 500
    
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
    app.run(debug=True, host='0.0.0.0', port=5000)
