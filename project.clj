(defproject keras-to-cortex "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-alpha14"]
                 [thinktopic/compute "0.1.0-SNAPSHOT"
                  :exclusions [org.clojure/clojurescript
                               org.clojure/tools.reader
                               org.clojure/clojure]]
                 [net.mikera/imagez "0.11.0"
                 :exclusions [net.mikera/clojure-utils]]
                 [net.mikera/core.matrix "0.57.0"
                 :exclusions [org.clojure/clojure
                              net.mikera/clojure-utils]]
                 [net.mikera/vectorz-clj "0.45.0"
                  :exclusions [org.clojure/clojure
                               net.mikera/core.matrix]]
                 [thinktopic/think.image "0.4.1"
                  :exclusions [net.mikera/imagez
                               org.clojure/clojure]]
                 [thinktopic/cortex-keras "0.1.0-SNAPSHOT"
                  :exclusions [org.clojure/clojure]]
                 [com.taoensso/nippy "2.12.2"]
                 [net.mikera/core.matrix "0.57.0"
                  :exclusions [org.clojure/clojure]]
                 ;; transitive dep of keras, have to follow its setup to get
                 ;; native lib deps, then wrap correctly.
                 #_[thinktopic/hdf5 "0.1.0-SNAPSHOT"]])
