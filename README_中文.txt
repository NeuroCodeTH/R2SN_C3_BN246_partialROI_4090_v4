C3 卒中患者 BN246 剩余 ROI 结构相似性网络构建（4090 服务器，无 Slurm）
版本：BN246_remaining_roi_zero_v4.1

【本版本与旧 BN246 的唯一区别（计算层面）】
1. 不再设置 “任意 ROI 病灶比例 >5% 则整名患者剔除” 的门槛。
2. 先在 R2SN 配准目标 MNI 1mm 网格内，用患者病灶从 Brainnetome 246 个 ROI 中扣除体素。
3. 对每一个有剩余体素的 ROI，继续调用官方 R2SN 的 PyRadiomics 参数 Params.yaml 提取影像组学特征；即使残余体素少于200也尝试提取，并标记 LOW_VOXELS_EXTRACTED。
4. ROI 若被病灶完全覆盖（--margin-mm 0），该节点在最终 246×246 网络的整行整列赋值0。
5. 如果有剩余体素，但 PyRadiomics 提取失败、存在无效特征或区域特征标准化后退化，则该 ROI 网络行列设为 NaN，不冒充病理性0。
6. 各项有效 ROI 特征按官方 R2SN 的定义：对每一项特征，在“全部有效 ROI”内 min-max 标准化；再对区域特征向量计算 Pearson 相关；最后填回 246×246 网络矩阵。
7. 保留原本 N4 + ANTs SyN 注册与固定空间 mask 逻辑，输入原始 T1 不复制不修改。
8. 全部耗时计算以被试并行方式运行，不依赖 Slurm；独立患者文件夹、断点续跑、逐患者日志。
9. 和旧方案分开 tag，不覆盖任何已完成的 BN210、BN246 或 QC。

【强烈建议主分析 --margin-mm 0】
  此时 “100% 病灶覆盖 ROI => 0” 与用户所定义的规则严格一致。
  如果想与既往 1 mm 病灶边缘排除设置一致，可 --margin-mm 1 --tag lesionaware_BN246_remainingROI_margin1_v4。
  注意 margin=1 时可能出现病灶未覆盖100%但加上排除边缘后 ROI 已无可用体素，编码 EMPTY_AFTER_MARGIN_ZERO，和 FULLY_LESIONED_ZERO 分开保留标记。

【输入目录】
  T1        /data02/tanghui/C3_T1/<前缀>/<001-0001>/*T1w*.nii.gz
  对照表    /data02/tanghui/C3_T1/code_id_all.txt
  病灶      /data02/tanghui/C3_lesion_newversion/C3_img_aftercheck/LesionMNI1mm/<数字ID>_Lesion2MNI1mmNL.nii.gz
  匹配CSV  /data02/tanghui/C3_R2SN/scripts/C3_MNI_manifest.csv
  结果      /data02/tanghui/C3_R2SN/<数字ID>/lesionaware_BN246_remainingROI_v4/

【运行前先确认旧批次已停止！】
  pgrep -af 'run_parallel.py|run_one.py'
  不要启动旧 BN246 严格 QC 批次和新 v4 批次同时处理 9828 人，避免CPU/内存相互抢占。

【解压】
  conda activate r2sn
  cd /data02/tanghui/C3_R2SN/scripts
  unzip R2SN_C3_BN246_partialROI_4090_v4.zip
  cd R2SN_C3_BN246_partialROI_4090_v4

【验证环境】
  python -c 'import R2SN, ants, nibabel, scipy, numpy, pandas, radiomics, SimpleITK; print("Environment OK")'
  python selftest.py

【如匹配表已经存在，可直接运行。若不存在】
  python make_manifest.py

【先测 3 人】
  python -u run_parallel.py \
    --workers 2 --threads-per-worker 2 --limit 3 \
    --tag lesionaware_BN246_remainingROI_v4 \
    --nroi 246 --margin-mm 0 --min-voxels 200 \
    --registration-mask fixed --keep-qc-images

  测试某一名曾被排除患者（例如 5429791731）：
  python -u run_parallel.py --id 5429791731 \
    --workers 1 --threads-per-worker 4 \
    --tag lesionaware_BN246_remainingROI_v4 \
    --margin-mm 0 --min-voxels 200 \
    --registration-mask fixed --keep-qc-images

【确认成功后整个队列后台并行】
  bash start_all_4090.sh

  可自定义并行（例如 4名被试并行、每人2线程）：
  nohup python -u run_parallel.py \
    --workers 4 --threads-per-worker 2 \
    --nroi 246 --tag lesionaware_BN246_remainingROI_v4 \
    --margin-mm 0 --min-voxels 200 --registration-mask fixed \
    > /data02/tanghui/C3_R2SN/BN246_remainingROI_v4.log 2>&1 < /dev/null &

【监控】
  tail -f /data02/tanghui/C3_R2SN/BN246_remainingROI_v4.log
  python check_status.py
  # 单例 R2SN 信息 +（可选）热图：
  python inspect_network.py --id 5429791731 --plot

【结果目录（每名患者）】
  <数字ID>/lesionaware_BN246_remainingROI_v4/
    qc/ROI_lesion_overlap.csv               每ROI损伤比例、剩余体素、状态
    qc/run_summary.json                    状态、算法版本、特征有效/缺失/清零数
    qc/registered_T1_MNI.nii.gz            仅加 --keep-qc-images 保存
    qc/lesion_resampled_MNI.nii.gz         仅加 --keep-qc-images 保存
    qc/BN_lesionExcluded.nii.gz            仅加 --keep-qc-images 保存
    radiomics/pyradiomics_input_radiomics_features.csv   成功提取的实际ROI原始特征表
    radiomics/ROI_features_246.csv        246 ROI 特征表，损毁/失败的行保留空值
    radiomics/ROI_node_metrics.csv         节点描述性指标+每节点状态
    network/<数字ID>_network.npy           【主矩阵】246×246，毁损0，未能计算NaN
    network/<数字ID>_valid_roi_mask.npy    bool 246个节点有效状态
    network/<数字ID>_network_zeroFilled_DISPLAY_ONLY.npy 仅画图/预览，不可当全部真实0用于推断

【状态定义】
  SUCCESS：所有未清零节点的影像组学特征已成功提取，完整网络有意义。
  SUCCESS_WITH_MISSING_NODES：有一个或多个非空ROI影像组学特征未能提取；主矩阵含NaN。
  FAILED：本次任务失败，查看 /data02/tanghui/C3_R2SN/lesionaware_logs/ 对应日志。

【正式科研使用注意】
  - ROI损伤越大，剩余组织的区域纹理特征与原 ROI 整体纹理特征的可比性越低。
  - 完全毁损节点的0是结构性零（人为编码“失去节点”），不是 Pearson=0。建模中使用 zero_rois/lesion_fraction/valid_mask 等指标。
  - 网络稀疏度受病灶大小强烈影响；结局关联应控制病灶体积、病灶侧别、基线NIHSS等，并做损伤阈值敏感性分析。
  - 对照组以及其他队列必须采取一致的归一化和网络处理原则，建议在完整 ROI 网络之外单独分析幸存节点子图。
  - 需要先用 FSLeyes/MRIcroGL 核验 registered_T1_MNI 与 lesion_resampled_MNI 解剖重合，MNI 1mm 文件名不能证明真正对齐。
  - 如果发生 100% 破坏的ROI很多，请将这类零编码与未受损节点的拓扑量分开汇报。

【测试范围】
  核心数学构造经过 synthetic 自测（完全损毁ROI、局部病灶、计算失败ROI、原始Pearson公式等价）。
  但无法在此容器获取你4090服务器上的真实T1和ANTs依赖，未完成真实患者影像端到端验证。
