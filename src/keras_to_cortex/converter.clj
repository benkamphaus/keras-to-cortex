(ns keras-to-cortex.converter
  (:require [think.cortex.keras.core :as keras]
            [taoensso.nippy :as nippy]
            [clojure.core.matrix :as m])
  (:import [java.io DataInputStream DataOutputStream])
  (:use clojure.repl))

(comment
  ;; dev example
  (require 'keras-to-cortex.converter)
  (in-ns 'keras-to-cortex.converter)

  (def converted-model-uri "resources/simple.cnn")

  (def model
    (keras/load-sidecar-and-verify "python/simple.json"
                                   "python/simple.h5"
                                   "python/simple_output.h5"))

  ;; save model
  (with-open [w (clojure.java.io/output-stream converted-model-uri)]
    (nippy/freeze-to-out! (DataOutputStream. w) model))

  ;; load model (sanity test)
  (def restored-model
    (with-open [r (clojure.java.io/input-stream converted-model-uri)]
      (nippy/thaw-from-in! (DataInputStream. r))))

  ;; quick sanity check that weights on last layer are equal.
  (let [spot-check (comp :weights last butlast)]
    (m/e== (spot-check model) (spot-check restored-model)))
  ;; should return `true`

  ;; full check on weights
  (let [full-check #(remove nil? (map :weights %))]
    (map m/e== (full-check model) (full-check restored-model)))
  ;; should return list with only `true`
)
