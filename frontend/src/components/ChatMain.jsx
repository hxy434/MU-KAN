import React, { useEffect, useRef, useState } from 'react';
import ChatBubble from './ChatBubble';
import { Spin, Upload, Button, message, Modal, Row, Col, Card, Image } from 'antd';
import { UploadOutlined, SendOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import axios from 'axios';

const imgTypeLabel = (key, t) => {
  if (!key) return '';
  if (key.includes('original')) return t('original');
  if (key.includes('probability')) return t('probability');
  if (key.includes('binary')) return t('mask');
  if (key.includes('overlay')) return t('overlay');
  if (key.includes('distribution')) return t('stats');
  return '';
};

export default function ChatMain({ session, loading: loadingProp, onAnalyze, model }) {
  const { t } = useTranslation();
  const chatRef = useRef();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [inputBubbles, setInputBubbles] = useState([]);
  const [particleModalVisible, setParticleModalVisible] = useState(false);
  const [selectedParticle, setSelectedParticle] = useState(null);
  const [particleDetails, setParticleDetails] = useState(null);

  // 自动滚动到底部
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [session, loading, inputBubbles]);

  useEffect(() => {
    setFile(null);
    setPreview(null);
    setInputBubbles([]);
  }, [session?.id]);

  // 处理粒子点击事件
  const handleParticleClick = async (particleId, sessionId) => {
    try {
      setSelectedParticle(particleId);
      setParticleModalVisible(true);
      
      // 调用API获取粒子详情
      const formData = new FormData();
      formData.append('particle_id', particleId);
      formData.append('session_id', sessionId);
      
      const response = await axios.post('/api/particle-details', formData);
      
      if (response.data.success) {
        setParticleDetails(response.data.particle);
      } else {
        message.error(response.data.error || '获取粒子详情失败');
        setParticleModalVisible(false);
      }
    } catch (error) {
      console.error('获取粒子详情失败:', error);
      message.error('获取粒子详情失败');
      setParticleModalVisible(false);
    }
  };

  // 关闭粒子详情模态框
  const handleParticleModalClose = () => {
    setParticleModalVisible(false);
    setSelectedParticle(null);
    setParticleDetails(null);
  };

  // 获取当前会话的所有图片
  const getCurrentImages = () => {
    if (!session || !session.rounds) return [];
    
    // 只获取最新一轮的分析结果
    const latestRound = session.rounds[session.rounds.length - 1];
    if (!latestRound || !latestRound.ai || !latestRound.ai.results) return [];
    
    const images = [];
    const ai = latestRound.ai;
    
    // 按顺序添加图片
    const imageTypes = [
      { key: 'original', label: t('original') },
      { key: 'probability', label: t('probability') },
      { key: 'binary', label: t('mask') },
      { key: 'overlay', label: t('overlay') },
      { key: 'distribution', label: t('stats') },
      { key: 'contour', label: t('contour_overlay') },
      { key: 'numbered', label: t('numbered_image'), isClickable: true },
      { key: 'scale_detection', label: t('scale_detection') }
    ];
    
    imageTypes.forEach(({ key, label, isClickable }) => {
      if (ai.results[key]) {
        images.push({
          src: ai.results[key],
          label,
          isClickable,
          sessionId: ai.results.session_id,
          particleDetails: ai.results.particle_details
        });
      }
    });
    
    return images;
  };

  // 获取当前会话的报告
  const getCurrentReport = () => {
    if (!session || !session.rounds) return '';
    
    // 只获取最新一轮的报告
    const latestRound = session.rounds[session.rounds.length - 1];
    if (!latestRound || !latestRound.ai || !latestRound.ai.results) return '';
    
    return latestRound.ai.results.report || '';
  };

  // 获取最新输入的图片名称
  const getLatestImageName = () => {
    if (!session || !session.rounds) return '';
    
    // 只获取最新一轮的图片名称
    const latestRound = session.rounds[session.rounds.length - 1];
    if (!latestRound || !latestRound.user) return '';
    
    return latestRound.user.filename || '';
  };

  // 获取比例尺信息
  const getScaleInfo = () => {
    if (!session || !session.scale_info) return null;
    
    const s = session.scale_info;
    if (s.pixel_to_real_ratio) {
      return {
        scale_length_pixels: s.scale_length_pixels,
        scale_length_real: s.scale_length_real,
        unit: s.unit,
        pixel_to_real_ratio: s.pixel_to_real_ratio
      };
    }
    return null;
  };

  // 组装对话流（用于上传和输入显示）
  const bubbles = [];
  if (session && session.rounds) {
    // 只显示最新一轮的用户输入信息
    const latestRound = session.rounds[session.rounds.length - 1];
    if (latestRound && latestRound.user) {
      bubbles.push({
        type: 'text',
        content: t('choose_file') + ': ' + (latestRound.user.filename || ''),
        isUser: true,
        time: latestRound.user.time,
        avatar: session.preview
      });
      bubbles.push({
        type: 'text',
        content: t('model') + ': ' + (latestRound.user.model || ''),
        isUser: true,
        time: latestRound.user.time,
        avatar: session.preview
      });
    }
  }
  // 用户输入气泡
  bubbles.push(...inputBubbles);

  // 上传图片
  const handleUpload = ({ file }) => {
    setFile(file);
    setPreview(URL.createObjectURL(file));
    setInputBubbles([{ type: 'text', content: t('choose_file') + ': ' + file.name, isUser: true }]);
  };

  // 分析
  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setInputBubbles(bubs => [...bubs, { type: 'text', content: t('analyzing'), isUser: true }]);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);
    try {
      const res = await axios.post('/api/analyze', formData);
      setInputBubbles([]);
      setFile(null);
      setPreview(null);
      if (onAnalyze) onAnalyze(res.data, file);
    } catch (e) {
      
    }
    setLoading(false);
  };

  const currentImages = getCurrentImages();
  const currentReport = getCurrentReport();
  const scaleInfo = getScaleInfo();
  const latestImageName = getLatestImageName();

  return (
    <div className="main-content" ref={chatRef} style={{ height: '100vh', paddingBottom: 32, display: 'flex', flexDirection: 'column' }}>
      {!loading && currentImages.length === 0 && (
        <div className="empty-state">{t('new_analysis') || 'Please upload image and analyze'}</div>
      )}
      
      {/* 隐藏用户输入气泡 */}
      {/* {bubbles.map((b, i) => (
        <ChatBubble 
          key={i} 
          {...b} 
          onParticleClick={handleParticleClick}
        />
      ))}
      
      {preview && (
        <ChatBubble type="image" content={preview} isUser={true} />
      )} */}
      
                    {/* 分析结果展示区域 */}
        {currentImages.length > 0 && (
          <div style={{ marginTop: 24, padding: '0 0' }}>
            <Row gutter={24} style={{ alignItems: 'flex-start' }}>
                           {/* 左侧图片区域 */}
              <Col span={12}>
                <Card 
                  title={`Analysis Results - ${latestImageName}`}
                  style={{ height: 'fit-content' }}
                  bodyStyle={{ padding: '16px' }}
                >
                                 <Row gutter={[12, 12]}>
                   {currentImages.map((img, index) => (
                     <Col span={12} key={index}>
                       <div style={{ textAlign: 'center' }}>
                         <Image
                           src={img.src}
                           alt={img.label}
                           style={{ 
                             borderRadius: 8, 
                             width: '100%', 
                             height: '120px', 
                             objectFit: 'contain',
                             cursor: 'pointer',
                             backgroundColor: '#f5f5f5'
                           }}
                           preview={{ mask: '点击放大' }}
                         />
                        <div style={{ 
                          color: '#1976d2', 
                          fontWeight: 500, 
                          fontSize: 12, 
                          marginTop: 4,
                          textAlign: 'center'
                        }}>
                          {img.label}
                        </div>
                      </div>
                    </Col>
                  ))}
                </Row>
              </Card>
            </Col>
            
                                       {/* 右侧报告区域 */}
              <Col span={12}>
                <Card 
                  title="Analysis Report" 
                  style={{ height: 'fit-content' }}
                  bodyStyle={{ padding: '16px', maxHeight: '600px', overflowY: 'auto' }}
                >
                {/* 比例尺信息 */}
                {scaleInfo && (
                  <div style={{ marginBottom: 16, padding: 12, background: '#f5f5f5', borderRadius: 8 }}>
                                         <h4 style={{ margin: '0 0 8px 0', color: '#1976d2' }}>Scale Information</h4>
                    <div style={{ fontSize: 14, lineHeight: 1.6 }}>
                                             <div>Scale Length (pixels): {scaleInfo.scale_length_pixels}</div>
                       <div>OCR Detected Physical Length: {scaleInfo.scale_length_real} {scaleInfo.unit}</div>
                       <div>Pixel to Physical Unit Ratio: {scaleInfo.pixel_to_real_ratio} {scaleInfo.unit}</div>
                    </div>
                  </div>
                )}
                
                                 {/* 分析报告 */}
                 {currentReport && (
                   <div>
                                           <h4 style={{ margin: '0 0 12px 0', color: '#1976d2' }}>Detailed Analysis Results</h4>
                     <pre style={{ 
                       background: '#f6f8fa', 
                       borderRadius: 8, 
                       fontSize: 13, 
                       padding: 12, 
                       margin: 0, 
                       color: '#222',
                       whiteSpace: 'pre-wrap',
                       wordBreak: 'break-word'
                     }}>
                       {currentReport}
                     </pre>
                   </div>
                 )}
                
                {!currentReport && !scaleInfo && (
                                     <div style={{ textAlign: 'center', color: '#999', padding: '40px 0' }}>
                     No analysis report available
                   </div>
                )}
              </Card>
            </Col>
          </Row>
        </div>
      )}
      
      {loading || loadingProp ? <Spin size="large" style={{ margin: '40px auto', display: 'block' }} /> : (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 24, maxWidth: 520, padding: '0 0' }}>
          <Upload
            showUploadList={false}
            accept="image/*"
            beforeUpload={file => { handleUpload({ file }); return false; }}
          >
            <Button icon={<UploadOutlined />} size="large" style={{ borderRadius: 20 }}>{t('upload')}</Button>
          </Upload>
          <Button
            type="primary"
            icon={<SendOutlined />}
            size="large"
            style={{ borderRadius: 20 }}
            disabled={!file || loading}
            onClick={handleAnalyze}
          >
            {t('analyze')}
          </Button>
        </div>
      )}
      
      {/* 粒子详情模态框 */}
      <Modal
                 title={`Particle ${selectedParticle} Details`}
        open={particleModalVisible}
        onCancel={handleParticleModalClose}
        footer={null}
        width={500}
      >
        {particleDetails && (
          <div style={{ padding: '20px 0' }}>
                         <div style={{ marginBottom: '15px' }}>
               <strong>Particle ID:</strong> {particleDetails.id}
             </div>
             <div style={{ marginBottom: '15px' }}>
               <strong>Area:</strong> {particleDetails.area}
             </div>
             <div style={{ marginBottom: '15px' }}>
               <strong>Diameter:</strong> {particleDetails.diameter}
             </div>
             <div style={{ marginBottom: '15px' }}>
               <strong>Circularity:</strong> {particleDetails.circularity}
             </div>
          </div>
        )}
      </Modal>
    </div>
  );
} 
