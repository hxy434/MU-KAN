import React from 'react';
import { Card, Image, Avatar } from 'antd';
import { RobotOutlined, UserOutlined } from '@ant-design/icons';

export default function ChatBubble({ type, content, isUser, time, avatar, label, sessionId, particleDetails, onParticleClick }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: isUser ? 'row' : 'row-reverse',
      alignItems: 'flex-end',
      marginBottom: 18,
      width: '100%',
      animation: 'fadeIn 0.5s',
    }}>
      <div style={{ margin: isUser ? '0 14px 0 0' : '0 0 0 14px' }}>
        <Avatar
          size={40}
          src={isUser && avatar ? avatar : undefined}
          icon={isUser ? <UserOutlined /> : <RobotOutlined />}
          style={{ background: isUser ? '#e3f0ff' : 'linear-gradient(135deg, #1976d2 0%, #26d0ce 100%)', color: isUser ? '#1976d2' : '#fff' }}
        />
      </div>
      <div style={{
        maxWidth: '70%',
        alignSelf: isUser ? 'flex-start' : 'flex-end',
      }}>
        <Card
          style={{
            borderRadius: isUser ? '18px 18px 6px 18px' : '18px 18px 18px 6px',
            background: isUser
              ? 'linear-gradient(90deg, #e3f0ff 0%, #f5faff 100%)'
              : 'linear-gradient(90deg, #1976d2 0%, #26d0ce 100%)',
            color: isUser ? '#222' : '#fff',
            boxShadow: isUser ? '0 2px 12px #b3d1ff33' : '0 2px 16px #1976d244',
            fontSize: 16,
            padding: 0,
            margin: 0,
            border: 'none',
            minWidth: 80,
            minHeight: 36,
            wordBreak: 'break-all',
            transition: 'box-shadow 0.2s',
          }}
          bodyStyle={{ padding: type === 'image' ? 0 : '16px 20px' }}
        >
          {type === 'text' && <span>{content}</span>}
          {type === 'image' && (
            <div style={{ textAlign: 'center' }}>
              <Image
                src={content}
                alt="分析图片"
                style={{ borderRadius: 12, maxWidth: 320, transition: 'transform 0.2s', cursor: 'pointer' }}
                preview={{ mask: '点击放大' }}
              />
              {label && <div style={{ color: '#1976d2', fontWeight: 500, fontSize: 15, marginTop: 8 }}>{label}</div>}
            </div>
          )}
          {type === 'clickable_image' && (
            <div style={{ textAlign: 'center' }}>
              <div style={{ position: 'relative', display: 'inline-block' }}>
                <Image
                  src={content}
                  alt="粒子编号图片"
                  style={{ borderRadius: 12, maxWidth: 320, transition: 'transform 0.2s', cursor: 'pointer' }}
                  preview={{ mask: '点击放大' }}
                />
                <div style={{ 
                  position: 'absolute', 
                  top: 0, 
                  left: 0, 
                  right: 0, 
                  bottom: 0, 
                  background: 'rgba(0,0,0,0.1)', 
                  borderRadius: 12,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  opacity: 0,
                  transition: 'opacity 0.2s'
                }}
                onMouseEnter={(e) => e.target.style.opacity = 1}
                onMouseLeave={(e) => e.target.style.opacity = 0}
                onClick={() => {
                  if (sessionId && particleDetails) {
                    // 显示可点击的粒子列表
                    const particleIds = Object.keys(particleDetails).map(Number);
                    if (particleIds.length > 0) {
                      // 这里可以显示一个选择器，让用户选择要查看的粒子
                      // 为了简化，我们直接显示第一个粒子
                      // 注意：这里需要从父组件传递处理函数
                      console.log('Particle clicked:', particleIds[0], sessionId);
                    }
                  }
                }}
                >
                  <div style={{ 
                    background: 'rgba(0,0,0,0.8)', 
                    color: 'white', 
                    padding: '8px 16px', 
                    borderRadius: 20,
                    fontSize: 14,
                    fontWeight: 500
                  }}>
                    点击查看粒子详情
                  </div>
                </div>
              </div>
              {label && <div style={{ color: '#1976d2', fontWeight: 500, fontSize: 15, marginTop: 8 }}>{label}</div>}
            </div>
          )}
          {type === 'report' && <pre style={{ background: isUser ? '#f6f8fa' : 'rgba(255,255,255,0.12)', borderRadius: 8, fontSize: 15, padding: 12, margin: 0, color: isUser ? '#222' : '#fff' }}>{content}</pre>}
        </Card>
        {time && <div style={{ fontSize: 12, color: '#888', margin: isUser ? '4px 0 0 8px' : '4px 8px 0 0', textAlign: isUser ? 'left' : 'right' }}>{time}</div>}
      </div>
      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px);} to { opacity: 1; transform: none; } }
      `}</style>
    </div>
  );
} 