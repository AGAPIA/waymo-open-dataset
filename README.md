# What i did on top of the existing code:

- Most important: segmentation method to put each LIDAR point cloud point on a certain label category. For this we use a forked version of ADE20K model trained available here: https://github.com/AGAPIA/semantic-segmentation-pytorch 
- if you look in Tutorial folder, there is a pipeline process that does all the job for taking a folder of datasegments from Waymo and producing the output. 
- Each pipeline stage is described at the top of each file, the main file that aggregates everything is pipeline_all.py: Following we describe its parameters and how to control things:

```
 --cleanOutputPath /home/ciprian/WAYMOOUTPUTMIN   # Where to write the FULL output (rgb images, segmentation images for comparison, etc)
 --fullOutputPath /home/ciprian/WAYMOOUTPUT   # Where to write only the files needed (Explained below what are those files and what they contain)
 --scenesFile /home/ciprian/WAYMOscenesToSolve.txt  # The path to a file contianing on each line the path to a single segment folder that you want to be processed by the pipeline
 --forceRecompute 0    # Do you want to recompute things if the resurces are already on disk ?
 --DEBUG_MIN_FRAME 0   # Put some values here if you want to cut the process only between some frames (currently waymo has up to 199 frames on each segment)
 --DEBUG_MAX_FRAME 99999 
 ```

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
