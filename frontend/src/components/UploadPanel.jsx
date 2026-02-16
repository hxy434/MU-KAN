import React, { useState } from 'react';
import { Upload, message, Spin } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import axios from 'axios';

const { Dragger } = Upload;

export default function UploadPanel({ onResult, model }) {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleUpload = async (options) => {
    const { file } = options;
    setLoading(true);
    setPreview(URL.createObjectURL(file));
    const formData = new FormData();
    formData.append('file', file);
    if (model) formData.append('model', model);
    try {
      const res = await axios.post('/api/analyze', formData);
      const toUrl = (b64) => 'data:image/png;base64,' + b64;
      onResult({
        original: toUrl(res.data.original),
        mask: toUrl(res.data.mask),
        overlay: toUrl(res.data.overlay),
        contour: toUrl(res.data.contour),
        stats: toUrl(res.data.stats),
        report: res.data.report,
        analysisFile: res.data.analysisFile // 兼容分析文件下载
      });
    } catch (e) {
      message.error('分析失败，请重试');
    }
    setLoading(false);
  };

  return (
    <Spin spinning={loading}>
      <Dragger
        name="file"
        customRequest={handleUpload}
        showUploadList={false}
        accept="image/*"
        style={{ padding: 24, borderRadius: 8, background: '#fafbfc' }}
      >
        <p className="ant-upload-drag-icon">
          <InboxOutlined style={{ color: '#1890ff', fontSize: 48 }} />
        </p>
        <p className="ant-upload-text">{t('upload') || '点击或拖拽图片到此处上传'}</p>
        {preview && (
          <img
            src={preview}
            alt="预览"
            style={{ marginTop: 16, maxWidth: 200, borderRadius: 8, boxShadow: '0 2px 8px #eee' }}
          />
        )}
      </Dragger>
    </Spin>
  );
} 