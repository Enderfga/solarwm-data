# H3 I2VA：给 Kimi 的 prompt

以下是 2026-10-05 最后补修使用的三段 prompt。早期批次使用过较早版本。
直接修改 `configs/kimi_prompt_h3_system.txt`、`configs/kimi_prompt_h3.txt`、
`configs/kimi_prompt_h3_post_images.txt` 即可，下次运行读取更新后的文本。
运行记录会保存实际使用的 prompt；修改文件不会影响已有合格结果的回填。

发送顺序：system → user 主文本 → 按时间排列的全片图片和时间戳 → 图片后的提醒。
全片每秒一张，最多 64 张，不读取音频。超过长度的输入先切分。

只返回四个字符串字段：`perspective_type`、`visual_style`、
`scene_description`、`scene_dynamics`。静态描述最多 80 个英文词，
独立背景运动最多 30 个词；保留主体外观，排除主体动作和相机动作。
灯光、屏幕、反射变化不写进动态描述。程序检查格式和规则，异常样本再看图片处理。

## System message

```text
Inspect every supplied video image. Return only the four string fields perspective_type, visual_style, scene_description, scene_dynamics. Describe only clearly confirmed physical content. Scene description: 1–3 sentences and at most 80 English words, static appearance only. Dynamics: empty or 1–2 sentences and at most 30 words, confirmed independent background displacement only. Exclude principal and observer actions, isolated near-edge body parts, camera/control prose, overlays, inferred identity, location, food, function and material. Never include any light, lamp, signal, screen, advertisement, illumination, reflection, shadow or glow event in dynamics. Omit uncertainty rather than guessing. Use first_person for setting-focused scenes without an external principal; uncertain is reserved for unresolved roles. Eligible independent motion also includes falling rain or snow, flowing water, physical flames and smoke, and wind-driven plants.
```

## User text before all images

```text
Inspect every supplied image of the complete video. Return one JSON object containing exactly four string fields: perspective_type, visual_style, scene_description, scene_dynamics. No explanation or extra keys.

perspective_type: first_person, third_person, or uncertain. Use third_person only for an external principal clearly and consistently followed as the subject. An incidental pedestrian does not establish a principal. Use first_person when the physical setting is the focus and no external principal is established. Use uncertain only when the evidence cannot distinguish these roles reliably.
visual_style: photographic live action; three-dimensional computer-generated imagery; two-dimensional animation; or mixed or uncertain visual medium. Decide from the images, not from the dataset or an unusual setting.

scene_description: 1–3 short English sentences, at most 80 words. State only clearly visible setting, objects and subject appearance. Prefer broad visible shape/color over guessed category. Static posture belongs here. No actor activity, camera/observer action, temporal sequence, or viewing directions. Keep lighting only as stable visible appearance here; never describe lighting changes.

scene_dynamics: an empty string or 1–2 short English sentences, at most 30 words. Include only clearly established independent background motion: incidental people walking, traffic moving, animals moving, flowing water, falling rain or snow, physical flames or smoke, and wind-driven plant movement. Require displacement relative to fixed scene objects, not camera movement. Omit every principal subject action and observer action. Never mention any lamp, light, signal, screen, advertisement, illumination, reflection, shadow, glow, optical flicker, or their changes in this field. Physical flames and smoke remain eligible when their independent movement is clearly visible; they are not excluded as lighting effects. These are deliberately excluded even when a change might be real.

Omit uncertain details completely. Never guess food, material, object function, location, brand, identity, or an exact count from weak evidence. A cone-shaped object is not wheat or pastry without clear evidence; a pink cloth is not a hat merely because it is near a head. Do not infer parked vehicles, water bodies, or digital displays from context. Omit isolated near-edge hands, fingers, arms, observer body parts and shadows without an established external actor. Exclude HUD, maps, scores, subtitles, watermarks, icons, control overlays and image artifacts. Do not narrate frames, sequence order, principal/control actions, audio or camera movement. Retain clearly established independent background motion; do not replace it with static presence.
```

## Final user text after all images

```text
All images are now supplied. Return concise grounded JSON. Static appearance only in scene_description (maximum 80 words). Confirmed independent background motion only in scene_dynamics (maximum 30 words); empty is allowed when none is established. Never include lamps, lighting, signals, screens, advertisements, reflections, shadows, glow or their changes in dynamics. Omit uncertain object labels, observer body parts, principal actions, controls, overlays and camera prose. Do not guess a pink cloth is a hat or cone-shaped items are food or wheat. Retain clearly observed falling rain or snow, flowing water, physical flames or smoke, and wind-driven plant motion. Use first_person for setting focus without an external principal; uncertain only for unresolved roles.
```

## 在 data-engine 中运行

两步都加 `--caption-format h3`。输入 JSONL 每行包含唯一的 `sample_id`、
`video_path`、`meta_path`，可选 `output_relpath`；路径可以是绝对路径或相对输入清单的路径。
源记录需含 `caption` 或已归档的 `static_scene_description`，尚未添加 H3 字段。
按清单逐条处理，不额外根据 `kept` 跳过。H3 模式保留旧 caption 和已有 VLM 指标，不重复打分。

```bash
export end_user=junchuang
python3 scripts/kimi_caption.py \
  --caption-format h3 \
  --manifest /absolute/input.jsonl \
  --output-dir /absolute/new-caption-run \
  --endpoint http://127.0.0.1:8000/v1 --workers 4

python3 scripts/kimi_materialize.py \
  --caption-format h3 \
  --manifest /absolute/input.jsonl \
  --caption-run /absolute/new-caption-run \
  --output-root /absolute/new-h3-overlay
```

每个服务只运行一个调度程序，最多同时 4 个请求。程序保存实际 prompt、模型信息和每次返回，
检查正常结束及 `prompt_tokens >= max(2000, 200 * frame_count)`，确认图片确实送入模型。
不合格响应携带失败内容重试一次；请求超时或连接异常时暂停新请求，确认服务请求结束后再启动。
H3 清单中任何样本仍不合格，命令都会返回失败并在汇总中记录，补齐后才能发布。
本地程序复用正式任务的 prompt、验证和模板生成逻辑；集群租约及跨运行补修由调度层负责。

回填写入独立的新目录：旧 `caption` 归档为 `static_scene_description`，新增
`h3_i2va`、`h3_prompt` 和标注来源信息。先发布完整 `meta.jsonl` 和
`META_JSONL_READY.json`，编码端即可读 `h3_prompt` 开始工作；之后回填逐片文件，
全部读回验收后写 `COMPLETE.json`。不合格结果保留错误记录，回填不会悄悄漏掉样本。
编码时读 `h3_prompt`，原始视频的绝对路径在 `h3_caption_provenance.source_video_path`。

## 写入每个 clip 的 H3 prompt

将 Kimi 返回的四个字段组合成下面的 H3 模板。`prompt.txt` 内容为 `h3_prompt` 加换行：

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] {visual_style}. {scene_description} {nonempty_scene_dynamics}

overall_soundscape: N/A

non_diegetic_music: N/A
```

不推断音频，两个音频字段固定为 `N/A`。
