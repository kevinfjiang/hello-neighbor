module Main (main) where

import Lib

main :: IO()
main = do
  searchBase <- readVecs getFVec "data/siftsmall_base.fvecs"
  queries <- readVecs getFVec "data/siftsmall_query.fvecs"

  let target = queries !! 3
      top = head searchBase

      ms = euclideanSpace $ take 3000 searchBase
      model = pInitLaesa ms 500
      pred1 = pPredict model top
      pred2 = pPredict model target

  print "In base search"
  print top
  print pred1
  print "Out base search"
  print target
  print pred2

  return ()


