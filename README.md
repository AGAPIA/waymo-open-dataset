# What i did on top of the existing code:

- Most important: segmentation method to put each LIDAR point cloud point on a certain label category. For this we use a forked version of ADE20K model trained available here: https://github.com/AGAPIA/semantic-segmentation-pytorch 
- if you look in Tutorial folder, there is a pipeline process that does all the job for taking a folder of datasegments from Waymo and producing the output. 
- Each pipeline stage is described at the top of each file, the main file that aggregates everything is pipeline_all.py the parameters description in the file to understand how to control things. What we have now are:

- options to produce very high-res point clouds by storing point cloud in float space
- produce motion frames by aggreating motion data (Cars and pedestrians) with environment
- clear noise using either KnnStatistical (best method now) or Voxelization
- debugging frames capabilities to simplify things
- parallization options on top of Ray and multi-GPU

Some other very useful scripts:

- To sort by pedestrians motion importance, i.e. how dense are scenes containing pedestrians we implemented a script in tutorial/scenesPedestrianInfoExtractor.py . You can use that to get a folder and output a csv file with sorted by "importance" the scenes.
- To play with point cloud visualization, conversion from Carla to Open3D ply formats, play with Open3D vizualization in place over point clouds use tutorial/PointcloudDemo.py script


 
### IMPORTANT NOTES: 
 - The output segmentation labels and colors are in CARLA space.
 - This folder contains already prebuilt files for Tensorflow 2.1 (which works with Cuda 10.1 as the segmentation fork). If you want to use a different folder, just follow the original instructions to install it again.


 

# Waymo Open Dataset

The Waymo Open Dataset is comprised of high-resolution sensor data collected by Waymo self-driving cars in a wide variety of conditions. We’re releasing this dataset publicly to aid the research community in making advancements in machine perception and self-driving technology.

## Website

To read more about the dataset and access it, please visit [https://www.waymo.com/open](https://www.waymo.com/open).

## Contents

This code repository contains:

* Definition of the dataset format
* Evaluation metrics
* Helper functions in TensorFlow to help with building models

Please refer to the [Quick Start](docs/quick_start.md).

## License
This code repository (excluding third_party) is licensed under the Apache License, Version 2.0.  Code appearing in third_party is licensed under terms appearing therein.

The Waymo Open Dataset itself is licensed under separate terms. Please visit [https://waymo.com/open/terms/](https://waymo.com/open/terms/) for details.  Code located at third_party/camera is licensed under a BSD 3-clause copyright license + an additional limited patent license applicable only when the code is used to process data from the Waymo Open Dataset as authorized by and in compliance with the Waymo Dataset License Agreement for Non-Commercial Use.  See third_party/camera for details.

## Citation
    @misc{sun2019scalability,
      title={Scalability in Perception for Autonomous Driving: Waymo Open Dataset},
      author={Pei Sun and Henrik Kretzschmar and Xerxes Dotiwalla and Aurelien Chouard and Vijaysai Patnaik and Paul Tsui and James Guo and Yin Zhou and Yuning Chai and Benjamin Caine and Vijay Vasudevan and Wei Han and Jiquan Ngiam and Hang Zhao and Aleksei Timofeev and Scott Ettinger and Maxim Krivokon and Amy Gao and Aditya Joshi and Yu Zhang and Jonathon Shlens and Zhifeng Chen and Dragomir Anguelov},
      year={2019},
      eprint={1912.04838},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

## Dataset Metadata
The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">Waymo Open Dataset: An autonomous driving dataset</code></td>
  </tr>
  <tr>
    <td>alternateName</td>
    <td><code itemprop="alternateName">Waymo Open Dataset</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/waymo-research/waymo-open-dataset</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://github.com/waymo-research/waymo-open-dataset</code></td>
  </tr>
    <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://www.waymo.com/open</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">The Waymo Open Dataset is comprised of high-resolution sensor data collected by Waymo self-driving cars in a wide variety of conditions. We’re releasing this dataset publicly to aid the research community in making advancements in machine perception and self-driving technology.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Waymo</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/Waymo</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Waymo Dataset License Agreement for Non-Commercial Use (August 2019)</code></td>
          </tr>
          <tr>
            <td>url</td>
            <td><code itemprop="url">https://waymo.com/open/terms/</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
</table>
</div>
