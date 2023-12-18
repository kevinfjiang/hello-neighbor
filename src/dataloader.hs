{-# LANGUAGE ScopedTypeVariables #-}
module DataLoader (readVecs, getFVec, getIVec) where

import Data.Int(Int32)
import Control.Monad(replicateM)

import Data.Binary.Get
import Data.ByteString.Lazy as BL
import Data.ByteString as BS

-- Gets a singular vector given dimension size
getFVec :: Int -> Get [Float]
getFVec d = do
  d <- replicateM d getFloatle
  _ <- getInt32le  -- terminating character
  return d

getIVec :: Int -> Get [Int32]
getIVec d = do
  d <- replicateM d (getInt32le)
  _ <- getInt32le  -- terminating character
  return d

-- Lazily reads the vectors from an .fvecs/.ivecs file
readVecs :: (Int -> Get m) -> FilePath -> IO [m]
readVecs getDimVec filename  = do
  contents <- BL.readFile filename
  let (d, rest) = runGet ((,) <$> getInt32le <*> getRemainingLazyByteString) contents
  return $ incrementalRead rest (getDimVec $ fromIntegral d)

-- Incremental read, source: https://hackage.haskell.org/package/binary-0.8.9.1/docs/Data-Binary-Get.html
incrementalRead :: forall m. BL.ByteString -> Get m -> [m]
incrementalRead input0 getFunc = go decoder input0
  where
    decoder = runGetIncremental getFunc

    go :: Decoder m -> BL.ByteString -> [m]
    go (Done leftover _consumed res) input = res : (go decoder $ newBytes)
      where newBytes = BL.append (BL.fromStrict leftover) input
    go (Partial k) input = go (k.takeHeadChunk $ input) (dropHeadChunk input)
    go (Fail _leftover _consumed msg) _input = error msg

-- idk why this is a strict bytestring, but it is
takeHeadChunk :: BL.ByteString -> Maybe BS.ByteString
takeHeadChunk lbs = case BL.uncons lbs of
  Just (h, _) -> Just $ BS.singleton h
  _ -> Nothing

dropHeadChunk :: BL.ByteString -> BL.ByteString
dropHeadChunk lbs = case BL.uncons lbs of
  Just (_, r) -> r
  _ -> BL.empty


apiuse :: IO()  -- TODO delete
apiuse = do
  v <- readVecs getIVec "data/siftsmall_groundtruth.ivecs"
  print $ ( Prelude.head v)
  return ()