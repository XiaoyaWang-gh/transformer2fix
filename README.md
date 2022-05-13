# transformer2fix
transformer2fix是一个用opennmt-py机器翻译框架实现的序列到序列模型，能够对Java的方法中出现的缺陷进行单行替换修复。
transformer2fix改进自模型sequencer，将编解码器类型从LSTM换成transformer的编解码器，并对其他超参数进行了大量修改。
训练集为CodRep和Bugs2fix。
测试集为defects4j。
迭代了2w个steps的transformer2fix在训练集上达到了83%的验证精度，2.0的验证困惑度。
用sequencer和transformer2fix分别对defects4j的101个单行替换缺陷进行修复，补丁推理阶段均使用50作为束宽(beam size)，前者正确修复了11个缺陷，后者正确修复了19个缺陷。

