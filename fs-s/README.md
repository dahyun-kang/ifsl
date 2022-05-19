
# Attentive Squeeze Network for Few-Shot Segmentation
Target task: (the conventional) few-shot segmentation

<hr>

## :deciduous_tree: Download authors' checkpoints
Download the checkpoint folder `logs/` from this One Drive link [[here]](https://postechackr-my.sharepoint.com/:f:/g/personal/kjdr86_postech_ac_kr/Elpa78QxKmhCtaRP8eMJvOMBxDzdSewcrK8ZIVhFkyiUMw?e=OCuWEC).

The file structure should be as follows:


    ifsl-submission-cvpr22/
    ├── fs-s
    │   ├──  common/
    │   ├──  data/
    │   ├──  model/
    │   ├──  logs/
    │   ├──  main.py
    │   └──  README.md
    ├── fs-cs
    ├── environment.yml
    └──  README.md


## :fire: Training a model
```bash
python main.py --datapath YOUR_DATASET_DIR \
               --benchmark {pascal, coco} \
               --logpath YOUR_DIR_TO_SAVE_CKPT \
               --way 1 \
               --shot 1 \
               --fold {0, 1, 2, 3} \
               --backbone {resnet50, resnet101}
```
Training ASNet on Pascal-5<sup>i</sup> takes 3 days to converge with 2 TitanXPs, and 6 days for COCO-20<sup>i</sup> with 4 TitanXPs and a double size of batch.

## :pushpin: Quick start: Evaluating authors' checkpoints

```bash
python main.py --datapath YOUR_DATASET_DIR \
               --benchmark {pascal, coco} \
               --logpath asnet \
               --shot {1, 5} \
               --backbone {vgg16, resnet50, resnet101} \
               --fold {0, 1, 2, 3} \
               --eval
```

To evaluate the `asnet-pascal-1shot-resnet50-fold0` model, run:
```bash
python main.py --datapath YOUR_DATASET_DIR \
               --benchmark pascal \
               --logpath asnet \
               --shot 1 \
               --backbone resnet50 \
               --fold 0 \
               --eval
```


## :art: (The conventional) few-shot segmentation results
Click the link to download the correponding model checkpoint.
### Experimental results of ASNet on Pascal-5<sup>i</sup> datasets on the FS-S task.

<table>
  <tr>
    <td colspan="14" align="center">ASNet</td>
  </tr>
  <tr>
    <td>setup</td>
    <td colspan="6" align="center">1-way 1-shot</td>
    <td colspan="6" align="center">1-way 5-shot</td>
  </tr>
    <tr>
    <td>backbone</td>
    <td align="center">5<sup>0</td>
    <td align="center">5<sup>1</td>
    <td align="center">5<sup>2</td>
    <td align="center">5<sup>3</td>
    <td align="center"><strong>mIoU</td>
    <td align="center"><strong>FBIoU</td>
    <td align="center">5<sup>0</td>
    <td align="center">5<sup>1</td>
    <td align="center">5<sup>2</td>
    <td align="center">5<sup>3</td>
    <td align="center"><strong>mIoU</td>
    <td align="center"><strong>FBIoU</td>
  </tr>
  <tr>
    <td>V16</td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EQ9QNqTUwItPjyEa773_yr8BTp-DzPEcQwyb1l-N_Dc7hQ?e=XuHfiM">61.7</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EXDZNI43PTNNvk_p02h4RbsBLxDTC-RfQTPr2Q-O6u6t1Q?e=fkZDBD">66.7</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/ETPoSbtT8iFHh-skupDVu9gB30DYL3SQdHwnvXuBBXadZA?e=fdoolF">58.6</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EYT3S9Yc4oBBqpsIS9AhOrYB7mT_faVFrJEbnBXUXIzjAg?e=zgE3fd">55.3</a></td>
    <td align="center"><strong>60.6</td>
    <td align="center"><strong>73.2</td>
    <td align="center">66.5</td>
    <td align="center">69.6</td>
    <td align="center">63.0</td>
    <td align="center">60.5</td>
    <td align="center"><strong>64.9</td>
    <td align="center"><strong>76.5</td>
  </tr>
    <tr>
    <td>R50</td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/ERkH6HRIjcFLkgj733_PXCUBmYEeE-xSFdoMqfugg6iSXg?e=2AiziM">68.9</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EQP1PYNR0p9NrbFHFf-_jgoBv2J5aj0cbkAV1PtDOFgVcw?e=KdiUXz">71.7</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EWSvZk2UxzRPmqe5R7F2R2oBwM6qYNDer9rFVZ2HAtNKlg?e=AKqxBo">61.1</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EQhEyKzlKUxKskB9dTPhidQBxxVhxJ4pTO27ei473x3OsQ?e=lMCzsW">62.7</a></td>
    <td align="center"><strong>66.1</td>
    <td align="center"><strong>77.7</td>
    <td align="center">72.6</td>
    <td align="center">74.3</td>
    <td align="center">65.3</td>
    <td align="center">67.1</td>
    <td align="center"><strong>70.8</td>
    <td align="center"><strong>80.4</td>
  </tr>
    <tr>
    <td>R101</td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EUDZw1bHHl1Jo5KGjDhYlvoBPYZR0ZdPWhuzrrpKl6h5fQ?e=DcYZFo">69.0</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/ESPjUeucotRKul5PNw_oeD0BsbmkJD4uf-NEQgwxtc8GnA?e=OGVxcG">73.1</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/Ef5HIvqexmhHr4JC_Xfpkv0Ba3cRvqKCsAEe7TW_4pEcpw?e=lpcZJZ">62.0</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EQNgS4yDnF9CkbZWKEgsimoBxvQVRvlxKDKdYdHlUc9Ibg?e=cKzwa1">63.6</a></td>
    <td align="center"><strong>66.9</td>
    <td align="center"><strong>78.0</td>
    <td align="center">73.1</td>
    <td align="center">75.6</td>
    <td align="center">65.7</td>
    <td align="center">69.9</td>
    <td align="center"><strong>71.1</td>
    <td align="center"><strong>81.0</td>
  </tr>
</table>

### Experimental results of ASNet on COCO-20<sup>i</sup> datasets on the FS-S task.

<table>
  <tr>
    <td colspan="14" align="center">ASNet</td>
  </tr>
  <tr>
    <td>setup</td>
    <td colspan="6" align="center">1-way 1-shot</td>
    <td colspan="6" align="center">1-way 5-shot</td>
  </tr>
    <tr>
    <td>backbone</td>
    <td align="center">5<sup>0</td>
    <td align="center">5<sup>1</td>
    <td align="center">5<sup>2</td>
    <td align="center">5<sup>3</td>
    <td align="center"><strong>mIoU</td>
    <td align="center"><strong>FBIoU</td>
    <td align="center">5<sup>0</td>
    <td align="center">5<sup>1</td>
    <td align="center">5<sup>2</td>
    <td align="center">5<sup>3</td>
    <td align="center"><strong>mIoU</td>
    <td align="center"><strong>FBIoU</td>
  </tr>
  <tr>
    <td>R50</td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EdIdskGuuhVBkRPIZ_b1rFYBn50vsLYp0Cl01k0G0l0BAQ?e=2XWdbw">41.5</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EX-A4mHioA1Fq4Lox4TLaqIBj9sDg8vji5-0qVaOIecUvQ?e=ljE3wT">44.1</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/ESqA_QY6fT9IvooeYxSu6boB_qobJeL1PEE6Ft76LXl_JQ?e=kFgYMV">42.8</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EU4IUizjFANJhZuQ9Z6wwuwBp-5BcRwRWL7yPESPY8zDSg?e=iBiukm">40.6</a></td>
    <td align="center"><strong>42.2</td>
    <td align="center"><strong>68.8</td>
    <td align="center">47.6</td>
    <td align="center">50.1</td>
    <td align="center">47.7</td>
    <td align="center">46.4</td>
    <td align="center"><strong>47.9</td>
    <td align="center"><strong>71.6</td>
  </tr>
    <tr>
    <td>R101</td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EVk0Eh0o9jBKgL-BD173_akBfhFA1e2gl9AQg_mbSWfWCw?e=WebXNQ">41.8</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EQXf__zGMVZKvBb5tDBzOYsBlHbh7pNpUMmcY_Li46HTHA?e=xdF51o">45.4</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/Ece0mLFzQGxCor8hhywJVNIB2bvjevm1Ho7F_TuUGe1www?e=FT5i6h">43.2</a></td>
    <td align="center"><a href="https://postechackr-my.sharepoint.com/:u:/g/personal/kjdr86_postech_ac_kr/EYgOZr_OLatDi619lPZzCiABm-RBhEAQrTrO_plo5lhu6A?e=mGLDNg">41.9</a></td>
    <td align="center"><strong>43.1</td>
    <td align="center"><strong>69.4</td>
    <td align="center">48.0</td>
    <td align="center">52.1</td>
    <td align="center">49.7</td>
    <td align="center">48.2</td>
    <td align="center"><strong>49.5</td>
    <td align="center"><strong>72.7</td>
  </tr>
</table>
