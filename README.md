# 🔍 Évaluation des Dommages Véhicules — CNN & Transfer Learning

L'évaluation manuelle des dommages carrosserie pour les assurances est lente,
coûteuse, et sujette à la subjectivité. Ce projet automatise ce processus avec
du deep learning : une photo du véhicule endommagé → localisation du dommage
et classification de la sévérité, au niveau de précision d'un expert humain.

Deux modèles CNN entraînés par transfer learning sur VGG16 : l'un pour la
**localisation** (avant/arrière/côté), l'autre pour la **sévérité** (mineur/modéré/sévère).

---

## Pipeline de classification

```
  Photo véhicule
  (upload utilisateur)
          │
          │  Preprocessing
          ▼
  ┌──────────────────────────────────────────┐
  │  Preprocessing                           │
  │                                          │
  │  - Resize → 224×224 (format VGG16)       │
  │  - Normalisation [0,1]                   │
  │  - ImageDataGenerator augmentation :     │
  │    flip, rotation, zoom, brightness      │
  └──────────────┬───────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
  ┌───────────┐    ┌───────────────┐
  │ Modèle 1  │    │  Modèle 2     │
  │           │    │               │
  │ WHERE is  │    │  HOW SEVERE   │
  │ the damage│    │  is the damage│
  │           │    │               │
  │  - Avant  │    │  - Mineur     │
  │  - Arrière│    │  - Modéré     │
  │  - Côté   │    │  - Sévère     │
  │           │    │               │
  │ Acc: 79%  │    │  Acc: 71%     │
  └─────┬─────┘    └───────┬───────┘
        │                  │
        └────────┬──────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │     Résultat final           │
  │                              │
  │  Zone    : ARRIÈRE           │
  │  Sévérité: MODÉRÉE           │
  │  Confiance: 76%              │
  └──────────────────────────────┘
```

---

## Architecture du modèle — Transfer Learning VGG16

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_damage_classifier(num_classes: int, freeze_layers: int = 15):
    """
    VGG16 pré-entraîné sur ImageNet, fine-tuné pour la classification de dommages.
    """
    # Charger VGG16 sans la tête de classification originale
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Geler les couches basses (features génériques : bords, textures)
    # Laisser libres les couches hautes (features spécifiques : carrosserie)
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False

    # Nouvelle tête de classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)           # régularisation → éviter l'overfitting
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),   # LR faible pour fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Modèle 1 : localisation (3 classes)
location_model = build_damage_classifier(num_classes=3, freeze_layers=15)

# Modèle 2 : sévérité (3 classes)
severity_model  = build_damage_classifier(num_classes=3, freeze_layers=18)
```

---

## Data augmentation — dataset limité

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Augmentation agressive car dataset limité (Google Images + Stanford Cars)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.7, 1.3],   # variation éclairage → robustesse terrain
    shear_range=0.1,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    'data/damage_images/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
```

---

## Résultats & métriques

```
  Modèle Localisation (Avant / Arrière / Côté)
  ─────────────────────────────────────────────
  Accuracy    : 79.3%
  F1 (macro)  : 0.77
  Classe la plus difficile : "Côté" (plus grande variabilité visuelle)

  Modèle Sévérité (Mineur / Modéré / Sévère)
  ─────────────────────────────────────────────
  Accuracy    : 71.1%
  F1 (macro)  : 0.69
  Comparable à la précision humaine estimée (~70-75%)
  Classe la plus difficile : frontière Mineur/Modéré (subjectif)
```

---

## Ce que j'ai appris

Le **freeze partiel** des couches VGG16 est la décision la plus importante.
Si on gèle tout : le modèle ne s'adapte pas aux spécificités visuelles des dommages
de carrosserie. Si on libère tout : on risque le catastrophic forgetting des features
ImageNet sur un dataset trop petit. J'ai testé de `freeze_layers=10` à `freeze_layers=20`
— le sweet spot pour ce dataset était autour de 15 couches gelées.

L'autre difficulté : la **frontière mineur/modéré** est floue même pour des humains.
Deux experts peuvent noter le même dommage différemment. En ML, cette ambiguïté
se retrouve dans la matrice de confusion — et elle est normale. Ajouter une classe
"incertain" (avec threshold de confiance $<$ 60\%) serait la bonne approche en production.

---

*Projet réalisé dans le cadre de ma formation ingénieur — ENSET Mohammedia*
*Par **Abderrahmane Elouafi** · [LinkedIn](https://www.linkedin.com/in/abderrahmane-elouafi-43226736b/) · [Portfolio](https://my-first-porfolio-six.vercel.app/)*
