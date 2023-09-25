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

### A.2. Relative Pose Estimation

Relocnet: Continuous metric learning relocalisation using neural nets

Learning to localize in new environments from synthetic training data

#### Geometry-based

Map-free visual relocalization: Metric pose relative to a single image

#### Regression-based

Relocnet: Continuous metric learning relocalisation using neural nets

Learning to localize in new environments from synthetic training data

Demon: Depth and motion network for learning monocular stereo

Relative camera pose estimation using convolutional neural networks

Rpnet: An end-to-end network for relative camera pose estimation

Camera relocalization by computing pairwise relative poses using convolutional neural network

Distillpose: Lightweight camera localization using auxiliary learning

Extreme rotation estimation using dense correlation volumes

Wide-baseline relative camera pose estimation with directional learning

Map-free visual relocalization: Metric pose relative to a single image

Discovery of latent 3d keypoints via end-to-end geometric reasoning

Estimating 3-d rigid body transformations: A comparison of four major algorithms

## B. Visual Landmark Map

### B.1. Local Feature Extraction-then-Matching

#### Local feature extraction

Distinctive image features from scale-invariant key points

Speeded-up robust features (SURF)

Brisk: Binary robust invariant scalable keypoints

Brief: Binary robust independent elementary features

Orb: An efficient alternative to sift or surf

Superpoint: Self-supervised interest point detection and description

R2D2: Repeatable and reliable detector and descriptor

D2-Net: A trainable CNN for joint description and detection of local features

SEKD: Self-evolving keypoint detection and description

Learning deep local features with multiple dynamic attentions for large-scale image retrieval

Large-scale image retrieval with attentive deep local features

Discriminative and semantic feature selection for place recognition towards dynamic environments

DISK: Learning local features with policy gradient

Mtldesc: Looking wider to describe better

Real-time local feature with global visual information enhancement

D2-Net: A trainable CNN for joint description and detection of local features

ALIKE: Accurate and lightweight keypoint detection and descriptor extraction

Aliked: A lighter keypoint and descriptor extraction network via deformable transformation

Learning discriminative and transformation covariant local feature detectors

Rap-net: A region-wise and point-wise weighting network to extract robust features for indoor localization

L2-net: Deep learning of discriminative patch descriptor in euclidean space

Working hard to know your neighbor's margins: Local descriptor learning loss

Sosnet: Second order similarity regularization for local descriptor learning

Geodesc: Learning local descriptors by integrating geometry constraints

Contextdesc: Local descriptor augmentation with cross-modality context

Aslfeat: Learning local features of accurate shape and localization

#### Local feature matching

A linear time histogram metric for improved sift matching

Scalable nearest neighbor algorithms for high dimensional data

Good match exploration using triangle constraint

Robust point matching via vector field consensus

Locality preserving matching

Guided locality preserving feature matching for remote sensing image registration

Code: Coherence based decision boundaries for feature correspondence

Gms: Grid-based motion statistics for fast, ultra-robust feature correspondence

Matching images with multiple descriptors: An unsupervised approach for locally adaptive descriptor selection

Guided matching based on statistical optical flow for fast and robust correspondence analysis

Rejecting mismatches by correspondence function

Feature matching with bounded distortion

Regularization based iterative point match weighting for accurate rigid transformation estimation

Dual calibration mechanism based l2, p-norm for graph matching

Second- and high-order graph matching for correspondence problems

Feature correspondence via graph matching: Models and global optimization

Bb-homography: Joint binary features and bipartite graph matching for homography estimation

Handcrafted outlier detection revisited

Image correspondence with cur decomposition-based graph completion and matching

Hierarchical semantic image matching using cnn feature pyramid

Nm-net: Mining reliable neighbors for robust feature correspondences

Oanet: Learning two-view correspondences and geometry using order-aware network

Learning to find good correspondences

Acne: Attentive context normalization for robust permutation-equivariant learning

Superglue: Learning feature matching with graph neural networks

### B.2. Joint Local Feature Extraction-and-Matching

Ncnet: Neighbourhood consensus networks for estimating image correspondences

Efficient neighbourhood consensus networks via submanifold sparse convolutions

Dual-resolution correspondence networks

Cotr: Correspondence transformer for matching across images

Loftr: Detector-free local feature matching with transformers

Quadtree attention for vision transformers

Aspanformer: Detector-free image matching with adaptive span transformer

Tkwinformer: Top k window attention in vision transformers for feature matching

Occ2net: Robust image matching based on 3d occupancy estimation for occluded regions

### B.3. Pose Solver

A general solution to the p4p problem for camera with unknown focal length

Camera pose and calibration from 4 or 5 known 3d points

A stable direct solution of perspective-three-point problem

Exact and approximate solutions of the perspective-three-point problem

Linear n-point camera pose determination

Efficient linear solution of exterior orientation

Epnp: An accurate o(n) solution to the pnp problem

A robust o(n) solution to the perspective-n-point problem

A direct least-squares (dls) method for pnp

Revisiting the pnp problem: A fast, general and optimal solution

A simple, robust and fast method for the perspective-n-point problem

An efficient and accurate algorithm for the perspecitve-n-point problem

Fast and globally convergent pose estimation from video images

6-dof object pose from semantic keypoints

A consistently fast and globally optimal solution to the perspective-n-point problem

Mlpnp - a real-time maximum likelihood solution to the perspective-n-point problem

Leveraging feature uncertainty in the pnp problem

Combining points and lines for camera pose estimation and optimization in monocular visual odometry

Uncertainty-aware camera pose estimation from points and lines

Cpnp: Consistent pose estimator for perspective-n-point problem with bias elimination

### B.4. Further Improvements

Inloc: Indoor visual localization with dense matching and view synthesis

Scenesqueezer: Learning to compress scene for camera relocalization

Line as a visual sentence: Context-aware line descriptor for visual localization

Is this the right place? geometric-semantic pose verifcation for indoor visual localization

Pose correction for highly accurate visual localization in large-scale indoor spaces

Cross-descriptor visual localization and mapping

Superglue: Learning feature matching with graph neural networks

Loftr: Detector-free local feature matching with transformers

Gn-net: The gauss-newton loss for multi-weather relocalization

Meshloc: Mesh-based visual localization

Structure-from-motion revisited

Benchmarking 6dof outdoor visual localization in changing conditions

Robust image retrieval-based visual localization using kapture

## C. Point Cloud Map

### C.1. Geometry-based

Stereo camera localization in 3d lidar maps

Monocular camera localization in 3d lidar maps

Visual localization within lidar maps for automated urban driving

Monocular camera localization in prior lidar maps with 2d-3d line correspondences

Metric monocular localization using signed distance fields

Monocular direct sparse localization in a prior 3d surfel map

Gmmloc: Structure consistent visual localization with gaussian mixture models

### C.2. Learning-based

#### Cross-modal visual place recognition

Global visual localization in lidar-maps through shared 2d-3d embedding space

Spherical multi-modal place recognition for heterogeneous sensor systems

i3dloc: Image-to-range cross-domain localization robust to inconsistent environmental conditions

Attention-enhanced cross-modal localization between spherical images and point clouds

I2p-rec:Recognizing images on large-scale point cloud maps through bird's eye view projections

#### Cross-modal relative pose regression

Cmrnet: Camera to lidar-map registration

Hypermap: Compressed 3d map for monocular camera registration

Poses as queries: Image-to-lidar map localization with transformers

### Cross-modal matching-based localization

2d3d-matchnet: Learning to match keypoints across 2d image and 3d point cloud

Cmrnet++: Map and camera agnostic monocular visual localization in lidar maps

Deepi2p: Image-to-point cloud registration via deep classification

Corri2p: Deep image-to-point cloud registration via dense correspondence

Efghnet: A versatile image-to-point cloud registration network for extreme outdoor environment

I2d-loc: Camera localization via image to lidar depth flow

End-to-end 2d-3d registration between image and lidar point cloud for vehicle localization

Neural reprojection error: Merging feature learning and camera pose estimation

Gn-net: The gauss-newton loss for multi-weather relocalization

Back to the feature: Learning robust camera localization from pixels to pose

## D. Vectorized HD Map

Tm3loc: Tightly-coupled monocular map matching for high precision vehicle localization

High definition map for automated driving: Overview and analysis map

Mapping and localization using gps, lane markings and proprioceptive sensors

Integration of gps, monocular vision, and high definition (hd) map for accurate vehicle localization

Coarse-to-fine semantic localization with hd map for autonomous driving in structural scenes

Self-localization based on visual lane marking maps: An accurate low-cost approach for autonomous driving

Semantic segmentation-based lane-level localization using around view monitoring system

A lane-level road marking map using a monocular camera

Monocular vehicle self-localization method based on compact semantic map

High precision vehicle localization based on tightly-coupled visual odometry and vector hd map

Coarse-to-fine visual localization using semantic compact map

Monocular localization in urban environments using road markings

Monocular localization in hd maps by combining semantic segmentation and distance transform

Vins-mono: A robust and versatile monocular visual-inertial state estimator

Video based localization for bertha

Monocular localization with vector hd map (mlvhm): A low-cost method for commercialivs

Bev-locator:An end-to-end visual semantic localization network using multi-view images

Egovm: Achieving precise ego-localization using lightweight vectorized maps

## E. Learnt Implicit Map

### E.1. Absolute Pose Regression

#### Single scene

Posenet: A convolutional network for real-time 6-dof camera relocalization

Do we really need scene-specific pose encoders?

Deep regression for monocular camera-based 6-dof global localization in outdoor environments

Image-based localization using hourglass networks

Modelling uncertainty in deep learning for camera relocalization

Image-based localization using lstms for structured feature correlation

Geometric loss functions for camera pose regression with deep learning

Improved visual relocalization by discovering anchor points

Atloc: Attention guided camera localization

Catiloc: Camera image transformer for indoor localization

Local supports global: Deep camera relocalization with sequence enhancement

Geometry-aware learning of maps for camera localization

#### Multiple scene

Extending absolute pose regression to multiple scenes

Learning multi-scene absolute pose regression with transformers

Coarse-to-fine multi-scene pose regression with transformers

### E.2. Scene Corrdinate Regression

Scene coordinate regression forests for camera relocalization in rgb-d images

Multi-output learning for camera relocalization

Exploiting uncertainty in regression forests for accurate camera relocalization

Dsac - differentiable ransac for camera localization

Visual camera re-localization from rgb and rgb-d images using dsac

Hierarchical scene coordinate classification and regression for visual localization

Visual localization via few-shot scene region classification

Learning camera localization via dense scene matching

Kfnet: Learning temporal camera relocalization using kalman filtering

Learning less is more - 6d camera localization via 3d surface regression

Vs-net: Voting with segmentation for visual localization

Sanet: Scene agnostic network for camera localization

#### Scene-specific SCR

Dsac - differentiable ransac for camera localization

Visual camera re-localization from rgb and rgb-d images using dsac

Learning less is more - 6d camera localization via 3d surface regression

Expert sample consensus applied to camera re-localization

Hierarchical scene coordinate classification and regression for visual localization

Visual localization via few-shot scene region classification

Vs-net: Voting with segmentation for visual localization

Visual localization by learning objects-of-interest dense match regression

Large scale joint semantic re-localisation and scene understanding via globally unique instance coordinate regression

Hscnet++: Hierarchical scene coordinate classification and regression for visual localization with transformer

#### Scene-agnostic SCR

Sanet: Scene agnostic network for camera localization

Learning camera localization via dense scene matching

Sacreg: Scene-agnostic coordinate regression for visual localization

Structure-from-motion revisited

Neumap: Neural coordinate mapping by auto-transdecoder for camera localization

Learning less is more - 6d camera localization via 3d surface regression

D2s: Representing local descriptors and global scene coordinates for camera relocalization

Real-time rgb-d camera relocalization

Expert sample consensus applied to camera re-localization

### E.3. Neural Radiance Field

Nerf: Representing scenes as neural radiance fields for view synthesis

#### As pose estimator

Gnerf: Gan-based neural radiance field without posed camera

Nope-nerf: Optimising neural radiance field with no pose prior

Locnerf: Monte carlo localization using neural radiance fields

Nerf-loc: Visual localization with conditional neural radiance field

Latitude: Robotic global localization with truncated dynamic low-pass filter in city-scale nerf

#### As data augmentation

Lens: Localization enhanced by nerf synthesis

Direct-posenet: Absolute pose regression with photometric consistency

Dfnet: Enhance absolute pose regression with direct feature matching
