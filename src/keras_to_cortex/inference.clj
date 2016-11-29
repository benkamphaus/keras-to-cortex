(ns keras-to-cortex.inference
  (:require [taoensso.nippy :as nippy]
            [think.compute.nn.train :as train]
            [think.compute.nn.description :as desc]
            [think.compute.nn.cpu-backend :as backend]
            [mikera.image.core :as img]
            [think.image.patch :as patch]
            [cortex.dataset :as ds]
            [think.compute.nn.train :as train])
  (:import [java.io DataInputStream DataOutputStream]))


(def image-shape
  (apply ds/create-image-shape [3 32 32]))

(def nn-model
  (future
    (desc/build-and-create-network
      (with-open [r (clojure.java.io/input-stream "resources/simple.cnn")]
        (nippy/thaw-from-in! (DataInputStream. r)))
      (backend/create-cpu-backend)
      1)))

(defn infer
  "Takes in input img, does munging necessary to get inference from nn-model, returns
  label/prediction probabilities as represented in softmax layer  produced by inference."
  [buf-img]
  (let [model   @nn-model
        resized (img/resize buf-img (:width image-shape) (:height image-shape))
        rgb-vec (patch/image->patch resized) ;; double is default 
        img-ds  (ds/create-in-memory-dataset {:data {:data [rgb-vec]
                                                     :shape image-shape}}
                                             [0])] ;; only one index, 0 (list of indexes in ds)
    (ffirst (train/run model img-ds [:data]))))

(defn read-image
  "Read image at img-path into buffered image."
  [img-path]
  (let [img-file (clojure.java.io/file img-path)]
    (img/load-image img-file)))

(comment
  ;; read in a couple of example images
  (def zazzy (read-image "resources/zazzy-cifar10.png"))
  (def car (read-image "resources/car-cifar10.png"))

  ;; run inference on them
  (infer zazzy)
  (infer car)

  ;; time cpu inference (for these small images w/simple model,
  ;; I see ~10 msec per inference.
  (time (infer zazzy))
)
