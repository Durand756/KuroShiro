# 🎨 Manga Style Converter

Un site web complet pour transformer vos dessins papier en style manga/BD professionnel en utilisant des algorithmes avancés de traitement d'image.

![Demo](https://img.shields.io/badge/Demo-Live-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.3+-red)

## ✨ Fonctionnalités

- 📤 **Upload facile** : Interface drag & drop pour JPG/PNG
- 🎯 **Traitement automatique** : Suppression du fond, amélioration des contours
- 🖤 **Style manga** : Conversion en noir & blanc pur avec trames
- 📱 **Responsive** : Compatible mobile, tablette et desktop
- 🔒 **Sécurisé** : Suppression automatique des fichiers après 1h
- ⚡ **Rapide** : Traitement en quelques secondes

## 🛠️ Technologies utilisées

- **Backend** : Flask (Python)
- **Traitement d'image** : OpenCV, Pillow, rembg
- **Frontend** : Bootstrap 5, HTML5, CSS3, JavaScript
- **Design** : Glassmorphisme, animations CSS

## 🚀 Installation rapide

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### 1. Cloner le projet
```bash
git clone https://github.com/votre-username/manga-converter.git
cd manga-converter
```

### 2. Créer un environnement virtuel
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Créer la structure des dossiers
```bash
mkdir -p templates static uploads processed
```

### 5. Lancer l'application
```bash
python app.py
```

🌐 **Accéder au site** : http://localhost:5000

## 📁 Structure du projet

```
manga-converter/
│
├── app.py                 # Application Flask principale
├── requirements.txt       # Dépendances Python
├── README.md             # Ce fichier
│
├── templates/
│   └── index.html        # Interface utilisateur
│
├── static/
│   └── style.css         # Styles personnalisés
│
├── uploads/              # Fichiers uploadés (créé automatiquement)
└── processed/            # Fichiers traités (créé automatiquement)
```

## 🎯 Comment utiliser

1. **Importer** : Cliquez sur "Choisir une image" et sélectionnez votre dessin
2. **Prévisualiser** : Vérifiez l'aperçu de votre image
3. **Transformer** : Cliquez sur "Transformer en style manga"
4. **Télécharger** : Récupérez votre dessin stylisé en haute résolution

## 🧠 Algorithmes de traitement

### Pipeline de transformation :
1. **Suppression du fond** avec rembg (IA)
2. **Conversion en niveaux de gris**
3. **Amélioration du contraste** (CLAHE)
4. **Seuillage adaptatif** pour du noir & blanc pur
5. **Débruitage** morphologique
6. **Ajout de trames manga** (optionnel)

### Paramètres optimisés :
- Seuillage adaptatif Gaussian (11x11)
- Morphologie : noyau 2x2
- Trames : points 4px, espacement 8px

## 🌍 Déploiement

### Sur Render (Gratuit)
1. Fork le projet sur GitHub
2. Connectez votre compte GitHub à Render
3. Créez un nouveau "Web Service"
4. Configurez :
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `gunicorn app:app`
   - **Environment** : Python 3

### Sur Railway (Gratuit)
1. Connectez votre repo GitHub
2. Railway détecte automatiquement Python
3. Ajoutez le fichier `Procfile` :
```
web: gunicorn app:app
```

### Variables d'environnement (optionnel)
```bash
export FLASK_ENV=production
export PORT=5000
```

## 🔧 Personnalisation

### Modifier les algorithmes de traitement
Éditez la fonction `process_image_to_manga()` dans `app.py` :

```python
# Ajuster le seuillage
binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv2.THRESH_BINARY, 11, 2)  # Modifier ces valeurs
```

### Personnaliser l'interface
- **Couleurs** : Modifiez les variables CSS dans `style.css`
- **Textes** : Éditez directement `index.html`
- **Fonctionnalités** : Ajoutez du JavaScript dans `index.html`

## 📋 Configuration avancée

### Limites de fichiers
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

### Nettoyage automatique
```python
cutoff_time = current_time - timedelta(hours=1)  # Modifier la durée
```

### Formats supportés
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # Ajouter d'autres formats
```

## 🐛 Dépannage

### Erreur "Module not found"
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Erreur OpenCV sur Linux
```bash
sudo apt-get update
sudo apt-get install libopencv-dev python3-opencv
```

### Erreur de mémoire
Réduisez la taille des images ou augmentez la RAM disponible.

### Port déjà utilisé
```bash
python app.py --port 8000  # Changer le port
```

## 📈 Améliorations futures

- [ ] Support des fichiers PDF
- [ ] Batch processing (plusieurs images)
- [ ] Filtres de style additionnels
- [ ] API REST pour intégration
- [ ] Mode sombre automatique
- [ ] Historique des conversions
- [ ] Compression d'images intelligente

## 🤝 Contribution

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.

## 🙏 Remerciements

- [OpenCV](https://opencv.org/) pour le traitement d'image
- [rembg](https://github.com/danielgatis/rembg) pour la suppression de fond
- [Bootstrap](https://getbootstrap.com/) pour l'interface utilisateur
- [Flask](https://flask.palletsprojects.com/) pour le framework web

## 📞 Support

- 📧 **Email** : support@manga-converter.com
- 🐛 **Issues** : [GitHub Issues](https://github.com/votre-username/manga-converter/issues)
- 💬 **Discord** : [Rejoignez notre serveur](https://discord.gg/manga-converter)

---

⭐ **N'hésitez pas à mettre une étoile si ce projet vous aide !**
