"""
refer to : https://github.com/serengil/deepface/blob/13a21fe306ee39567f7f0b15422f8a3c1ce656de/deepface/DeepFace.py#L721
"""
import numpy as np
import functions
import VGGFace


def verify(img1, img2, model=None, normalization='VGGFace'):
	"""
	This function verifies an image pair is same person or different persons.
	Parameters:
		img1_path, img2_path: exact image path, numpy array (BGR) or based64 encoded images could be passed. If you are going to call verify function for a list of image pairs, then you should pass an array instead of calling the function in for loops.
		e.g. img1_path = [
			['img1.jpg', 'img2.jpg'],
			['img2.jpg', 'img3.jpg']
		]
		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace or Ensemble
		distance_metric (string): cosine, euclidean, euclidean_l2
		model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times.
			model = DeepFace.build_model('VGG-Face')
	Returns:
		Verify function returns a dictionary. If img1_path is a list of image pairs, then the function will return list of dictionary.
		{
			"verified": True
			, "distance": 0.2563
		}
	"""
	img1_representation = represent(img = img1, model=model, normalization = normalization)
	img2_representation = represent(img = img2, model=model, normalization = normalization)

	#----------------------
	#find distances between embeddings
	distance = functions.findCosineDistance(img1_representation, img2_representation)
	distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)

	# threshold of VGGFace
	threshold = 0.40

	if distance <= threshold:
		identified = True
	else:
		identified = False

	resp_obj = {
		"verified": identified
		, "distance": distance
		, "threshold": threshold
	}

	#-------------------------
	return resp_obj

def represent(img, model, normalization = 'base'):
	"""
	This function represents facial images as vectors.
	Parameters:
		img: exact image path, numpy array (BGR) or based64 encoded images could be passed.
		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace.
		normalization (string): normalize the input image before feeding to model
	Returns:
		Represent function returns a multidimensional vector. The number of dimensions is changing based on the reference model. E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
	"""
	if model is None:
		model = VGGFace.loadModel()

	#decide input shape
	input_shape_x, input_shape_y = functions.find_input_shape(model)

	#detect and align
	img = functions.preprocess_face(img = img, target_size=(input_shape_y, input_shape_x))

	#detect and align
	#---------------------------------
	#custom normalization

	img = functions.normalize_input(img = img, normalization = normalization)

	#---------------------------------

	#represent
	if "keras" in str(type(model)):
		#new tf versions show progress bar and it is annoying
		embedding = model.predict(img, verbose=0)[0].tolist()
	else:
		#SFace is not a keras model and it has no verbose argument
		embedding = model.predict(img)[0].tolist()

	return embedding

