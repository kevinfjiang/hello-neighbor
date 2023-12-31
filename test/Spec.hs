module Main (main) where

import Test.Hspec
import Lib

main :: IO ()
main = hspec $ do
  describe "Laesa" $ do
    it "should initialize correctly" $ do
      let ms = euclideanSpace [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
      let laesa = initLaesa ms 2
      lNumBases laesa `shouldBe` 2

    it "should find closest neighbor" $ do
      let ms = euclideanSpace [[2.0, 3.0, 4.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
      let laesa = initLaesa ms 2
      let target = [1.0, 1.0, 1.0]
      let prediction = predict laesa target
      prediction `shouldBe` [2.0, 3.0, 4.0]

  describe "PLaesa" $ do
    it "should initialize correctly" $ do
      let ms = euclideanSpace [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
      let plaesa = pInitLaesa ms 2
      lNumBases plaesa `shouldBe` 2

    it "should find closest neighbor" $ do
      let ms = euclideanSpace [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
      let plaesa = pInitLaesa ms 2
      let target = [7.0, 8.0, 9.0]
      let prediction = pPredict plaesa target
      prediction `shouldBe` [4.0, 5.0, 6.0]

  describe "MetricSpace" $ do
    it "should compute distances correctly" $ do
      let ms = euclideanSpace [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
      let dist = mDist ms [1.0, 2.0, 3.0] [4.0, 5.0, 6.0]
      dist `shouldBe` 5.196152422706632
