# 第三方组件与许可声明

本项目的代码以 **MIT**（见根目录 `LICENSE`）发布。但它**随仓库再分发**了一个第三方
产出的二进制模型，另外运行时还会从公开 CDN 拉取第三方代码。按各自许可证的要求，
下面逐项说明来源、改动和许可。

---

## 1. 随仓库再分发的二进制

### `public/models/attractive.int4.onnx`（56.8 MB）

| 项 | 内容 |
|---|---|
| 上游模型 | [`dima806/attractive_faces_celebs_detection`](https://huggingface.co/dima806/attractive_faces_celebs_detection) |
| 其 base 模型 | [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k) |
| 许可证 | **Apache-2.0**（两个都是；上游模型卡的 `cardData.license` 字段核实过） |
| 本仓库的角色 | 非著作权人。这是**再分发**，不是原创产出 |

**我们对它做过的改动**（Apache-2.0 第 4(b) 条要求标注改动）：

1. **导出为 ONNX**，opset 17，LogitsOnly 包装（去掉 HuggingFace head 之外的输出），
   动态 batch 维、固定 224×224 输入；
2. **量化**：先动态 INT8，再用 `MatMulNBits` 对权重做 4-bit weight-only 量化
   （block size 32）；
3. **归一化外移**：输入预处理改为在调用方完成（`x/127.5 - 1`），
   模型本身只接收已归一化的张量 —— 从 HuggingFace 的 `AutoImageProcessor`
   配置里搬到了前端。

产出的文件（`attractive.int4.onnx`）是上述改动的结果，作为**衍生作品**同样适用
Apache-2.0。

导出与量化脚本在 `scripts/model/`，任何人可以按 `CLAUDE.md` 记的流程复现同样的文件。

---

## 2. 随仓库再分发的图片素材

`public/samples/` 下的 6 张图是首页「示例照片」按钮的素材（连缩略图共 12 个文件，
约 1.3 MB）。用途只有一个：让第一次来的访客不必先准备一张自己的正脸照，点一下就能
看到完整流程和结果长什么样。

它们全部取自 **Wikimedia Commons**，都是**真实人物的公开活动照** —— 这一点和
校准用的那 37 张公有领域官方肖像不同，所以在这里单独列清楚。

⚠️ 下表每一行都是**许可证义务**，不是致谢：CC BY 与 CC BY-SA 都要求署名，
CC BY-SA 还要求改编作品按**同一许可**再分发。换图、换裁剪、改文件名时这张表必须
同步改；页面另有一处可折叠的同内容署名区（`src/app.ts` 里的 `SAMPLES` 表），
两边要一起改。

**我们做过的改动**：**只有 `01` 做了裁剪** —— 它是竖幅全身/半身构图，脸在画面里太小，
用一个以检测器量到的虹膜中点为基准的头部框裁成头像构图（脚本 `tmp/samples/crop.py`，
不入库）。其余 5 张**都是原图整幅**，没有裁过。6 张一律缩放到最长边 ≤ 1280、
转成渐进式 JPEG、去掉元数据（`-strip`）；缩略图由入库的那张再缩到 260 px。
**除此之外没有任何改动** —— 没有美颜、没有换脸、没有生成式处理，也不是 AI 生成的图。

注意 `04` 与 `06` 的来源文件名里带 `(cropped)` 后缀，那是 **Commons 上游文件本身
就是裁剪版**，不是我们裁的。

| 入库文件 | 人物 | 作者 | 许可证 | 来源（Commons 原始文件页） |
|---|---|---|---|---|
| `01-liu-yifei.jpg` | 刘亦菲 | 刘亦菲吧官方 | [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/) | [File:Liu Yifei Portrait2.jpg](https://commons.wikimedia.org/wiki/File:Liu_Yifei_Portrait2.jpg) |
| `02-yang-mi.jpg` | 杨幂 | Mercury水星記·杨幂 | [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/) | [File:杨幂 ELLE30周年风尚大典（4）.jpg](https://commons.wikimedia.org/wiki/File:%E6%9D%A8%E5%B9%82_ELLE30%E5%91%A8%E5%B9%B4%E9%A3%8E%E5%B0%9A%E5%A4%A7%E5%85%B8_%EF%BC%884%EF%BC%89.jpg) |
| `03-emma-watson.jpg` | Emma Watson | David Shankbone | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | [File:Emma Watson, 2012.jpg](https://commons.wikimedia.org/wiki/File:Emma_Watson,_2012.jpg) |
| `04-henry-cavill.jpg` | Henry Cavill | ryanmorrisonjsy | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | [File:Henry Cavill-2665842 (cropped).jpg](https://commons.wikimedia.org/wiki/File:Henry_Cavill-2665842_(cropped).jpg) |
| `05-chris-hemsworth.jpg` | Chris Hemsworth | Gage Skidmore | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | [File:Chris Hemsworth by Gage Skidmore 2 (cropped).jpg](https://commons.wikimedia.org/wiki/File:Chris_Hemsworth_by_Gage_Skidmore_2_(cropped).jpg) |
| `06-li-xian.jpg` | 李现 | 李现_秃头姐妹站 | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | [File:Li Xian 3 (cropped).jpg](https://commons.wikimedia.org/wiki/File:Li_Xian_3_(cropped).jpg) |

**许可范围怎么划**：`04` 那张是 CC0，作者已放弃全部权利，署名只是出于礼貌。
`01` / `05` / `06` 三张是 **CC BY-SA**，条目里入库的 `*.jpg` 与 `*.thumb.jpg` 是原图的
**改编作品**（裁剪 + 缩放），按 share-alike 同样以 CC BY-SA（对应版本）提供；
它们和本项目 MIT 的**代码**是各自独立的作品，share-alike 不延伸到 `src/`。

**原始大图不入库**（放在 gitignored 的 `tmp/samples/orig/`，含每张的作者、许可、
来源记录）。要重新生成入库的那 12 个文件，见 `CLAUDE.md` 里记的 ImageMagick 命令。

---

## 3. 运行时不随仓库分发、从公开 CDN 加载

这些不在仓库里，也不由本项目再分发 —— 浏览器直接从各家的官方 CDN 取。列在这里
是为了说明依赖关系，不构成对本项目的许可授予。

| 组件 | 版本 | 来源 | 许可证 |
|---|---|---|---|
| MediaPipe Tasks Vision（wasm + `face_landmarker.task`） | 1.0.1 | jsDelivr / Google 官方模型库 | Apache-2.0 |
| onnxruntime-web（wasm） | 1.29.0 | jsDelivr | MIT |
| `onnxruntime-web` 的 JS 包 | 1.29.0 | 走 npm 依赖，打包进产物 | MIT |

人脸检测模型 `face_landmarker.task`（3.6 MB）由 Google 的
`storage.googleapis.com/mediapipe-models/` 提供，许可同样是 Apache-2.0。

---

## 4. 设计参考（未使用其代码）

`Blueturboguy07/freeharmony`（AGPL-3.0）只在设计阶段作为几何指标的**参考**读过，
**一行代码都没有进本仓库** —— 理由见 `CLAUDE.md`：AGPL-3.0 与本项目的
「可公开再分发」目标冲突。本项目的几何引擎（`src/engine/`）是独立实现的。

---

## 5. 本项目产出

`src/`、`scripts/`、`index.html` 等全部原创代码，以及几何指标的校准数据
（`src/engine/calibration.ts` 里的均值/标准差/理想值），按 MIT 发布。

校准样本是 37 张美国国会官方肖像 —— **公有领域**（美国联邦政府雇员职务作品），
且**图片本身不入库**。详见 `scripts/calibration/README.md`。

---

# 附：Apache License 2.0 全文

以下为上面两款 Apache-2.0 组件所适用的许可证全文（Apache-2.0 第 4(a) 条要求
再分发时随附副本）。

                              Apache License
                        Version 2.0, January 2004
                     http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

   "License" shall mean the terms and conditions for use, reproduction,
   and distribution as defined by Sections 1 through 9 of this document.

   "Licensor" shall mean the copyright owner or entity authorized by
   the copyright owner that is granting the License.

   "Legal Entity" shall mean the union of the acting entity and all
   other entities that control, are controlled by, or are under common
   control with that entity. For the purposes of this definition,
   "control" means (i) the power, direct or indirect, to cause the
   direction or management of such entity, whether by contract or
   otherwise, or (ii) ownership of fifty percent (50%) or more of the
   outstanding shares, or (iii) beneficial ownership of such entity.

   "You" (or "Your") shall mean an individual or Legal Entity
   exercising permissions granted by this License.

   "Source" form shall mean the preferred form for making modifications,
   including but not limited to software source code, documentation
   source, and configuration files.

   "Object" form shall mean any form resulting from mechanical
   transformation or translation of a Source form, including but
   not limited to compiled object code, generated documentation,
   and conversions to other media types.

   "Work" shall mean the work of authorship, whether in Source or
   Object form, made available under the License, as indicated by a
   copyright notice that is included in or attached to the work
   (an example is provided in the Appendix below).

   "Derivative Works" shall mean any work, whether in Source or Object
   form, that is based on (or derived from) the Work and for which the
   editorial revisions, annotations, elaborations, or other modifications
   represent, as a whole, an original work of authorship. For the purposes
   of this License, Derivative Works shall not include works that remain
   separable from, or merely link (or bind by name) to the interfaces of,
   the Work and Derivative Works thereof.

   "Contribution" shall mean any work of authorship, including
   the original version of the Work and any modifications or additions
   to that Work or Derivative Works thereof, that is intentionally
   submitted to Licensor for inclusion in the Work by the copyright owner
   or by an individual or Legal Entity authorized to submit on behalf of
   the copyright owner. For the purposes of this definition, "submitted"
   means any form of electronic, verbal, or written communication sent
   to the Licensor or its representatives, including but not limited to
   communication on electronic mailing lists, source code control systems,
   and issue tracking systems that are managed by, or on behalf of, the
   Licensor for the purpose of discussing and improving the Work, but
   excluding communication that is conspicuously marked or otherwise
   designated in writing by the copyright owner as "Not a Contribution."

   "Contributor" shall mean Licensor and any individual or Legal Entity
   on behalf of whom a Contribution has been received by Licensor and
   subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   copyright license to reproduce, prepare Derivative Works of,
   publicly display, publicly perform, sublicense, and distribute the
   Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   (except as stated in this section) patent license to make, have made,
   use, offer to sell, sell, import, and otherwise transfer the Work,
   where such license applies only to those patent claims licensable
   by such Contributor that are necessarily infringed by their
   Contribution(s) alone or by combination of their Contribution(s)
   with the Work to which such Contribution(s) was submitted. If You
   institute patent litigation against any entity (including a
   cross-claim or counterclaim in a lawsuit) alleging that the Work
   or a Contribution incorporated within the Work constitutes direct
   or contributory patent infringement, then any patent licenses
   granted to You under this License for that Work shall terminate
   as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
   Work or Derivative Works thereof in any medium, with or without
   modifications, and in Source or Object form, provided that You
   meet the following conditions:

   (a) You must give any other recipients of the Work or
       Derivative Works a copy of this License; and

   (b) You must cause any modified files to carry prominent notices
       stating that You changed the files; and

   (c) You must retain, in the Source form of any Derivative Works
       that You distribute, all copyright, patent, trademark, and
       attribution notices from the Source form of the Work,
       excluding those notices that do not pertain to any part of
       the Derivative Works; and

   (d) If the Work includes a "NOTICE" text file as part of its
       distribution, then any Derivative Works that You distribute must
       include a readable copy of the attribution notices contained
       within such NOTICE file, excluding those notices that do not
       pertain to any part of the Derivative Works, in at least one
       of the following places: within a NOTICE text file distributed
       as part of the Derivative Works; within the Source form or
       documentation, if provided along with the Derivative Works; or,
       within a display generated by the Derivative Works, if and
       wherever such third-party notices normally appear. The contents
       of the NOTICE file are for informational purposes only and
       do not modify the License. You may add Your own attribution
       notices within Derivative Works that You distribute, alongside
       or as an addendum to the NOTICE text from the Work, provided
       that such additional attribution notices cannot be construed
       as modifying the License.

   You may add Your own copyright statement to Your modifications and
   may provide additional or different license terms and conditions
   for use, reproduction, or distribution of Your modifications, or
   for any such Derivative Works as a whole, provided Your use,
   reproduction, and distribution of the Work otherwise complies with
   the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
   names, trademarks, service marks, or product names of the Licensor,
   except as required for reasonable and customary use in describing the
   origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
   agreed to in writing, Licensor provides the Work (and each
   Contributor provides its Contributions) on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied, including, without limitation, any warranties or conditions
   of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
   PARTICULAR PURPOSE. You are solely responsible for determining the
   appropriateness of using or redistributing the Work and assume any
   risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
   whether in tort (including negligence), contract, or otherwise,
   unless required by applicable law (such as deliberate and grossly
   negligent acts) or agreed to in writing, shall any Contributor be
   liable to You for damages, including any direct, indirect, special,
   incidental, or consequential damages of any character arising as a
   result of this License or out of the use or inability to use the
   Work (including but not limited to damages for loss of goodwill,
   work stoppage, computer failure or malfunction, or any and all
   other commercial damages or losses), even if such Contributor
   has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
   the Work or Derivative Works thereof, You may choose to offer,
   and charge a fee for, acceptance of support, warranty, indemnity,
   or other liability obligations and/or rights consistent with this
   License. However, in accepting such obligations, You may act only
   on Your own behalf and on Your sole responsibility, not on behalf
   of any other Contributor, and only if You agree to indemnify,
   defend, and hold each Contributor harmless for any liability
   incurred by, or claims asserted against, such Contributor by reason
   of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

   To apply the Apache License to your work, attach the following
   boilerplate notice, with the fields enclosed by brackets "[]"
   replaced with your own identifying information. (Don't include
   the brackets!)  The text should be enclosed in the appropriate
   comment syntax for the file format. We also recommend that a
   file or class name and description of purpose be included on the
   same "printed page" as the copyright notice for easier
   identification within third-party archives.

Copyright 2019 The CryptoCorrosion Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
