# Results: fasterrcnn_grocery_run
_Generated: 2026-04-06 16:17:11_

## Run Configuration
| Parameter | Value |
|-----------|-------|
| Device | cuda |
| Epochs | 80 |
| Batch size | 2 |
| Image size | 640 |
| Best weights | `/content/EE3703 RPC/fasterrcnn_grocery_run/best.pt` |

## Dataset Split
| Split | Images | Original images | Boxes |
|------:|------:|----------------:|------:|
| Train | 384 | 85 | 2641 |
| Val | 66 | 0 | 541 |
| Test | 50 | 50 | 123 |

## Test Metrics
| Metric | Value |
|-------:|------:|
| mAP@0.50 | 0.8467 |
| mAP@0.50:0.95 | 0.4929 |
| Precision | 0.7569 |
| Recall | 0.8862 |
| F1 | 0.8165 |
| Score threshold | 0.60 |

## AP50 by Class
| Class | AP50 |
|------|-----:|
| Bottled Drink | 0.7978 |
| Canned Goods | 0.8004 |
| Fresh Produce | 0.9471 |
| Packaged Food | 0.8414 |

## Inference Benchmark
| Metric | Value |
|-------:|------:|
| Images | 50 |
| Mean latency (ms) | 46.80 |
| Std latency (ms) | 1.45 |
| Min latency (ms) | 44.34 |
| Max latency (ms) | 50.54 |
| FPS | 21.37 |

## Most Difficult Test Images
| Rank | Image | Errors | FP | FN | Wrong Class |
|----:|------|------:|---:|---:|------------:|
| 1 | self_stanley_orig/20260309_211741.jpg | 4 | 3 | 0 | 1 |
| 2 | self_stanley_orig/photo_2026-03-15 09.59.20.jpeg | 4 | 3 | 0 | 1 |
| 3 | self_girvin/IMG_9685_1_jpg.rf.Npf3kUGAjHwDxXwnoYFY.jpg | 3 | 2 | 0 | 1 |
| 4 | self_naveen/IMG_1514_JPG.rf.SJ7OM4wvv9jCKVvVpybx.JPG | 3 | 3 | 0 | 0 |
| 5 | self_naveen/IMG_1535_JPG.rf.A5CCjtsoV5J5vA9pWYBJ.JPG | 3 | 2 | 1 | 0 |
| 6 | self_naveen/IMG_1673_JPG.rf.U1xP6X6KOdwodEInpDfV.JPG | 3 | 2 | 1 | 0 |
| 7 | self_stanley_orig/20260309_213914.jpg | 3 | 2 | 0 | 1 |
| 8 | self_naveen/IMG_1536_JPG.rf.bo9CR7xqK7y7jrS7AfEi.JPG | 2 | 1 | 1 | 0 |
| 9 | self_stanley_add/photo_2026-03-26 23.11.16.jpeg | 2 | 1 | 1 | 0 |
| 10 | self_stanley_add/photo_2026-03-26 23.11.18.jpeg | 2 | 1 | 1 | 0 |
| 11 | self_stanley_add/photo_2026-03-26 23.11.22.jpeg | 2 | 0 | 1 | 1 |
| 12 | self_stanley_orig/photo_2026-03-15 09.59.17.jpeg | 2 | 1 | 0 | 1 |
| 13 | self_girvin/IMG_9695_jpg.rf.Da91ylYvyuWHoZiiCafu.jpg | 1 | 1 | 0 | 0 |
| 14 | self_naveen/IMG_1674_JPG.rf.Xb3OSBAvMjgeyyoBppmj.JPG | 1 | 0 | 1 | 0 |
| 15 | self_naveen/IMG_1676_JPG.rf.dV4VUJTf3jWDjSPKVoFM.JPG | 1 | 0 | 1 | 0 |
| 16 | self_naveen/IMG_1685_JPG.rf.6u2hQCDQ5toNg2CL6qAp.JPG | 1 | 1 | 0 | 0 |
| 17 | self_naveen/IMG_1686_JPG.rf.0nvUB2XBRkopkxHGN3wp.JPG | 1 | 1 | 0 | 0 |
| 18 | self_naveen/IMG_1687_JPG.rf.UUNihQdlLAq2CeKJSrw0.JPG | 1 | 1 | 0 | 0 |
| 19 | self_naveen/IMG_1688_JPG.rf.ZbEA9C00tkUgHe3F54LO.JPG | 1 | 1 | 0 | 0 |
| 20 | self_naveen/IMG_1694_JPG.rf.xZUO8K8gBPI5BKL7UQxP.JPG | 1 | 1 | 0 | 0 |
| 21 | self_stanley_add/photo_2026-03-26 23.11.09.jpeg | 1 | 1 | 0 | 0 |
| 22 | self_stanley_add/photo_2026-03-26 23.11.25.jpeg | 1 | 0 | 0 | 1 |
| 23 | self_stanley_orig/20260309_211704.jpg | 1 | 0 | 0 | 1 |
| 24 | self_stanley_orig/20260309_211810.jpg | 1 | 0 | 0 | 1 |
| 25 | self_stanley_orig/20260309_214122.jpg | 1 | 1 | 0 | 0 |

## Clean Test Examples
| Rank | Image | GT boxes | Pred boxes |
|----:|------|--------:|----------:|
| 1 | self_naveen/IMG_1668_JPG.rf.UZuP0W5g9I8dJdyL15SZ.JPG | 5 | 5 |
| 2 | self_girvin/IMG_9680_jpg.rf.Qj97J8ApkCCzzNS7qNeP.jpg | 3 | 3 |
| 3 | self_girvin/IMG_9700_jpg.rf.MojuwRZH3wmrMpBw1Df6.jpg | 3 | 3 |
| 4 | self_girvin/IMG_9701_jpg.rf.eLhSyr9TWVX4AjGlUvcY.jpg | 3 | 3 |
| 5 | self_naveen/IMG_1666_JPG.rf.VwWwRd3G2QNNia6NonRm.JPG | 3 | 3 |
| 6 | self_naveen/IMG_1693_JPG.rf.CdGOz2kImMZqbBWYlwaz.JPG | 3 | 3 |
| 7 | self_stanley_orig/20260309_211901.jpg | 3 | 3 |
| 8 | self_girvin/IMG_9683_jpg.rf.xKP659u3ZgU6rJshzbVj.jpg | 2 | 2 |