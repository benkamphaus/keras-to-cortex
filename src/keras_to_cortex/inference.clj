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

(def test-image
  (let [img-file (clojure.java.io/file "resources/zazzy-cifar10.png")]
    (img/load-image img-file)))

(comment
  (infer test-image)
)
