import torch
import numpy as np
import base64
import requests
import time
import os
import json
from io import BytesIO
from PIL import Image
from comfy.utils import ProgressBar

# 特效模板映射
TEMPLATE_MAPPING = {
    # 普通特效
    "解压捏捏": "squish",
    "转圈圈": "rotation",
    "载歌乐": "poke",
    "气球膨胀": "inflate",
    "分子扩散": "dissolve",
    
    # 单人特效
    "时光木马": "carousel",
    "爱你哟": "singleheart",
    "摇摆时刻": "dance1",
    "头号甩舞": "dance2",
    "星摇时刻": "dance3",
    "人鱼光耀": "mermaid",
    "学术加冕": "graduation",
    "巨幕追袭": "dragon",
    "财从天降": "money",
    
    # 单人或动物特效
    "魔法悬浮": "flying",
    "赠人玫瑰": "rose",
    "闪亮玫瑰": "crystalrose",
    
    # 双人特效
    "爱的抱抱": "hug",
    "唇齿相依": "frenchkiss",
    "双倍心动": "coupleheart"
}

class WanVideoCreateTask:
    """
    创建WAN视频生成任务节点
    输入图像和提示词，调用API创建视频生成任务
    返回任务ID供结果查询节点使用
    """
    @classmethod
    def INPUT_TYPES(cls):
        # 生成特效下拉选项（包含用户特效名称）
        template_options = ["无特效"] + list(TEMPLATE_MAPPING.keys())
        
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "一只猫在草地上奔跑",
                    "description": "提示词（使用特效时此字段无效）"
                }),
                "model": (["wan2.2-i2v-plus", "wanx2.1-i2v-turbo"], {"default": "wan2.2-i2v-plus"}),
                "resolution": (["480P", "1080P"], {"default": "1080P"}),
                "template": (template_options, {"default": "无特效"}),
                "negative_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "低分辨率,错误,最差质量,低质量,残缺",
                    "description": "负面提示词（不希望出现在视频中的内容）"
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("task_id",)
    FUNCTION = "create_task"
    CATEGORY = "视频生成/WAN"
    
    def tensor_to_base64(self, tensor):
        """将ComfyUI的IMAGE张量转换为Base64"""
        i = 255. * tensor[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

    def create_task(self, api_key, image, prompt, model, resolution, template, negative_prompt, seed=0):
        # 转换图像格式
        try:
            img_base64 = self.tensor_to_base64(image)
        except Exception as e:
            raise Exception(f"图像转换失败: {str(e)}")
        
        # 准备API请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"
        }
        
        # 准备请求体
        input_data = {
            "img_url": img_base64,
            "negative_prompt": negative_prompt
        }
        
        # 应用特效模板（如果选择）
        if template != "无特效":
            input_data["template"] = TEMPLATE_MAPPING[template]
        else:
            input_data["prompt"] = prompt
        
        payload = {
            "model": model,
            "input": input_data,
            "parameters": {
                "resolution": resolution,
                "seed": seed
            }
        }
        
        # 发送请求
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    return {"error": f"API错误({response.status_code}): {error_data.get('message', '未知错误')}"}
                except:
                    return {"error": f"API请求失败: {response.status_code} - {response.text}"}
            
            result = response.json()
            if "output" in result and "task_id" in result["output"]:
                return (result["output"]["task_id"],)
            else:
                return {"error": f"API返回格式错误: {json.dumps(result, ensure_ascii=False)}"}
        
        except requests.Timeout:
            return {"error": "API请求超时，请检查网络连接或稍后重试"}
        except Exception as e:
            return {"error": f"发生未预期错误: {str(e)}"}


class WanVideoPollResult:
    """
    WAN视频结果查询节点
    输入任务ID，定期查询API直到任务完成
    下载生成的视频文件并返回本地路径
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "task_id": ("STRING", {"description": "从创建任务节点获取的任务ID"}),
                "poll_interval": ("INT", {
                    "default": 10, 
                    "min": 5,
                    "max": 120,
                    "description": "查询API的时间间隔(秒)"
                }),
                "max_retries": ("INT", {
                    "default": 30, 
                    "min": 1,
                    "max": 100,
                    "description": "最大查询次数(约5分钟)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "poll_result"
    CATEGORY = "视频生成/WAN"
    OUTPUT_NODE = True  # 启用输出节点状态更新

    def poll_result(self, api_key, task_id, poll_interval, max_retries):
        url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        pbar = ProgressBar(max_retries)
        
        last_status = ""
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code != 200:
                    pbar.update(1, "查询失败，重试中...")
                    time.sleep(poll_interval)
                    continue
                
                data = response.json()
                status = data.get("output", {}).get("task_status", "UNKNOWN")
                
                # 更新进度条显示
                status_display = f"{status} ({attempt+1}/{max_retries})"
                pbar.update(1, status_display)
                last_status = status
                
                if status == "SUCCEEDED":
                    video_url = data["output"]["video_url"]
                    filepath = self.download_video(video_url)
                    return {"result": (filepath,)}
                elif status in ["FAILED", "CANCELED"]:
                    error_msg = data.get("message", "未知错误")
                    return {"error": f"任务失败: {error_msg}"}
                
            except requests.Timeout:
                pbar.update(1, "查询超时，重试中...")
            except Exception as e:
                pbar.update(1, f"查询错误: {str(e)}")
            
            # 等待下一次查询
            time.sleep(poll_interval)
        
        # 达到最大尝试次数仍未完成
        return {"error": f"任务超时未完成，最后状态: {last_status}"}

    def download_video(self, video_url):
        """下载视频到ComfyUI/output/wan_videos目录"""
        try:
            # 在ComfyUI根目录创建输出文件夹
            comfyui_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
            output_dir = os.path.join(comfyui_root, "output", "wan_videos")
            os.makedirs(output_dir, exist_ok=True)
            
            # 确保下载目录存在
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    return f"创建输出目录失败: {str(e)}"
            
            # 创建唯一文件名
            timestamp = int(time.time())
            filename = f"wan_video_{timestamp}.mp4"
            filepath = os.path.join(output_dir, filename)
            
            # 下载视频
            response = requests.get(video_url, stream=True, timeout=120)
            if response.status_code != 200:
                return f"视频下载失败: {response.status_code}"
            
            # 分块写入文件
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB chunks
            
            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # 如果需要，可以在此处添加下载进度显示
            
            return filepath
        
        except Exception as e:
            return f"视频下载过程中出错: {str(e)}"


# 将节点映射导出给ComfyUI
NODE_CLASS_MAPPINGS = {
    "WanVideoCreateTask": WanVideoCreateTask,
    "WanVideoPollResult": WanVideoPollResult
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoCreateTask": "创建WAN视频任务",
    "WanVideoPollResult": "获取WAN视频结果"
}