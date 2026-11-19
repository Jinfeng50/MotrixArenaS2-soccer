# Attribution and licensing

The Decider framework retains the original **MOS-Brain Contributors** copyright
and `GPL-3.0-or-later` identifiers. See [LICENSE](LICENSE) and [COPYRIGHT](COPYRIGHT).
Team modifications to the derived Decider code are distributed under the same terms.

The bundled `decider/interfaces/zmq_wrapper.py`, license text, and copyright notice
were obtained from [JasonZhw/sim_soccer2](https://github.com/JasonZhw/sim_soccer2)
at commit `1dce38430a651f67c90d19643be0f80c63d52c91`.
The wrapper's original path is `simulation/isaac_sim/src/common/zmq_wrapper.py`;
only formatting was changed. This attribution does not establish the repository
as an official MotrixArena S2 distribution.

The ONNX checkpoint in `gait/` was supplied with the team's submission.
Its training source, base-model provenance, and redistribution license were not
provided. **The code license is not a separate license grant for these weights.**
Confirm the checkpoint's redistribution terms before publishing it. Its export
debug strings have been removed without changing its computation or parameters.

Python dependencies retain their respective licenses. Simulator binaries, robot
assets, and third-party training frameworks are not vendored in this repository.

中文：保留 MOS-Brain 原有版权及 GPL-3.0-or-later 声明。新增通信封装来源及版本如上。
随交付包提供的步态模型缺少训练来源和独立再分发许可说明，发布权重前需核实；
不能仅凭代码的 GPL 声明推定模型权重许可。
