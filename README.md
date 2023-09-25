# visual-relocalization-papers
a paper list of visual re-localization algorithms

## A. Geo-tagged Frame Map

**Geo-tagged Frame Map** is composed of posed keyframes. The related visual relocalization algorithm can be classified into two categories: *Visual Place Recognition* and *Relative Pose Estimation*.

![VPR](https://github.com/jinyummiao/visual-relocalization/assets/38695019/b716fa14-6c96-4593-b3be-c256f45e32b2)

### A.1. Visual Place Recognition

#### Survey

Where is your place, visual place recognition? [paper](https://www.ijcai.org/proceedings/2021/0603.pdf)

Visual place recognition: A survey [paper](https://ieeexplore.ieee.org/document/7339473)

The revisiting problem in simultaneous localization and mapping: A survey on visual loop closure detection [paper](https://ieeexplore.ieee.org/document/9780121)

#### Global feature-based

Netvlad: CNN architecture for weakly supervised place recognition [paper](https://ieeexplore.ieee.org/document/7937898) [code(MATLAB)](https://ieeexplore.ieee.org/document/7937898) [code(PyTorch)](https://github.com/Nanne/pytorch-NetVlad)

Anyloc: Towards universal visual place recognition [arxiv](https://arxiv.org/abs/2308.00688) [code](https://github.com/AnyLoc/AnyLoc)

Histograms of oriented gradients for human detection

Vector of locally and adaptively aggregated descriptors for image feature representation

Spatial pyramid enhanced netvlad with weighted triplet loss for place recognition

Fine-tuning cnn image retrieval with no human annotation

Transvpr: Transformer-based place recognition with multi-level attention aggregation

Mixvpr: Feature mixing for visual place recognition

Real-time loop detection with bags of binary words

Bags of binary words for fast place recognition in image sequences

Biologically inspired mobile robot vision localization

Appearance-based place recognition for topological localization

Co-hog: A light-weight, compute-efficient, and training-free visual place recognition technique for changing environments

Aggregating local descriptors into a compact image representation

Esa-vlad: A lightweight network based on second-order attention and netvlad for loop closure detection

Seqnet: Learning descriptors for sequence-based hierarchical place recognition

Delta descriptors: Change-based place representation for robust visual localization

Leveraging deep visual descriptors for hierarchical efficient localization

Mixvpr: Feature mixing for visual place recognition

Patch-netvlad: Multi-scale fusion of locally-global descriptors for place recognition

#### Local feature-based 

Bocnf: Efficient image matching with bag of convnet features for scalable and robust visual place recognition

Place recognition with convnet landmarks: Viewpoint-robust, condition-robust, training-free

On the performance of convnet features for place recognition

Real-time loop detection with bags of binary words

Bags of binary words for fast place recognition in image sequences

Robust loop closure detection based on bag of superpoints and graph verification

Automatic vocabulary and graph verification for accurate loop closure detection

Loop-closure detection using local relative orientation matching

Discriminative and semantic feature selection for place recognition towards dynamic environments

FAB-MAP: Probabilistic localization and mapping in the space of appearance

Appearance-only SLAM at large scale with FAB-MAP 2.0

Automatic visual Bag-of-Words for online robot navigation and mapping

IBuILD: Incremental bag of binary words for appearance based loop closure detection

iBoW-LCD: An appearance-based loop-closure detection approach using incremental bags of binary words

Assigning visual words to places for loop closure detection

Probabilistic appearance-based place recognition through bag of tracked words

Hierarchical place recognition for topological mapping

Assigning visual words to places for loop closure detection

LiPo-LCD: Combining lines and points for appearance-based loop closure detection

LCD: Combining lines and points for appearance-based loop closure detection with the combination of points and lines based on information entropy

#### Semantic-based

X-View: Graph-based semantic multi-view localization

#### Sequence-based

SeqSLAM: Visual route-based navigation for sunny summer days and stormy winter nights

Fast-SeqSLAM: A fast appearance based place recognition algorithm

Seqnet: Learning descriptors for sequence-based hierarchical place recognition

Delta descriptors: Change-based place representation for robust visual localization

### A.2. Relative Pose Regression

#### Geometry-based

#### Learning-based

## B. Visual Landmark Map

### B.1. Local Feature Extraction-then-Matching

#### Local feature extraction

#### Local feature matching

### B.2. Joint Local Feature Extraction-and-Matching

### B.3. Pose Solver

### B.4. Further Improvements

## C. Point Cloud Map

### C.1. Geometry-based

### C.2. Learning-based

#### Cross-modal place recognition

#### Cross-modal relative pose regression

### Cross-modal matching-based localization

## D. Vectorized HD Map

## E. Learnt Implicit Map

### E.1. Absolute Pose Regression

#### Single scene

#### Multiple scene

### E.2. Scene Corrdinate Regression

#### Single scene

#### Multiple scene

### E.3. Neural Radiance Field

#### As pose solver

#### As data augmentor
