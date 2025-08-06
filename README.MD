# ComfyUI WAN视频生成插件

此插件允许在ComfyUI中通过API调用阿里云百炼（WAN）生成高质量视频。

## 功能特点
- 支持图生视频（Image-to-Video）
- 提供中文特效选择界面
- 支持480P和1080P分辨率
- 支持提示词和负面提示词
- 自动轮询任务状态并下载结果
- 视频保存到ComfyUI输出目录

## 安装
1. 将此插件文件夹放入 `ComfyUI/custom_nodes/` 目录
2. 重启ComfyUI

## 使用说明
### 节点说明
1. **创建WAN视频任务**
   - `api_key`: 阿里云百炼API密钥
   - `image`: 输入图像（支持ComfyUI图像节点）
   - `prompt`: 视频描述（使用特效时无效）
   - `model`: WAN模型版本（默认wan2.2-i2v-plus）
   - `resolution`: 视频分辨率
   - `template`: 视频特效模板
   - `negative_prompt`: 负面描述
   - `seed`: 随机种子（0表示随机）

2. **获取WAN视频结果**
   - `task_id`: 从创建任务节点获取的任务ID
   - `poll_interval`: 查询间隔（默认10秒）
   - `max_retries`: 最大查询次数（默认30次）

### 典型工作流
```
[加载图像] --> [创建WAN视频任务] --> [任务ID] --> [获取WAN视频结果] --> [视频路径]
                            ↑                         ↑
                       [API密钥]                   [API密钥]
```

## 注意事项
1. 需要有效的阿里云百炼API密钥
2. 视频生成可能需要3-5分钟时间
3. 视频文件保存在 `ComfyUI/output/wan_videos/` 目录
4. 使用特效时无需填写提示词

## 特效列表
| 中文名称 | 特效代码 |
|----------|----------|
| 魔法悬浮 | flying |
| 解压捏捏 | squish |
| 赠人玫瑰 | rose |
| ... | ... |

## 许可证
MIT License