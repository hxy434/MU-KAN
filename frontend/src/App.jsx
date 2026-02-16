import React, { useState } from 'react';
import { Layout, Dropdown, Menu, Button } from 'antd';
import { BulbOutlined, BulbFilled } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import Sidebar from './components/Sidebar';
import ChatMain from './components/ChatMain';
import './App.css';

const { Header, Sider, Content } = Layout;

const MODELS = [
  { key: 'MU-KAN', label: 'MU-KAN' },
  { key: 'YOLO', label: 'YOLO' },
];

export default function App() {
  const { t, i18n } = useTranslation();
  const [model, setModel] = useState('MU-KAN');
  const [history, setHistory] = useState(() => {
    const h = localStorage.getItem('analysis_history');
    if (!h) return [];
    // 兼容老结构
    const arr = JSON.parse(h);
    if (arr.length && !arr[0].rounds) {
      // 升级为多轮结构
      return arr.map(item => ({
        id: item.id,
        time: item.time,
        model: item.model,
        rounds: [{
          user: { filename: item.filename, model: item.model, time: item.time },
          ai: { results: item.results, time: item.time }
        }],
        preview: item.preview
      }));
    }
    return arr;
  });
  const [activeId, setActiveId] = useState(history[0]?.id || null);
  const [loading, setLoading] = useState(false);
  const [theme, setTheme] = useState(window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

  // 切换明暗模式
  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : 'light';
    setTheme(next);
    document.body.setAttribute('data-theme', next);
  };

  // 切换模型
  const handleModelChange = ({ key }) => setModel(key);

  // 新分析
  const handleNew = () => {
    setActiveId(null);
  };

  // 清空历史
  const handleClear = () => {
    setHistory([]);
    setActiveId(null);
    localStorage.removeItem('analysis_history');
  };

  // 选择历史
  const handleSelect = (id) => setActiveId(id);

  // 保存分析结果到历史（多轮）
  const onAnalyze = (aiResults, file) => {
    const now = new Date();
    let newHistory = [...history];
    let session;
    if (activeId) {
      // 追加到当前会话
      newHistory = newHistory.map(item => {
        if (item.id === activeId) {
          const round = {
            user: { filename: file?.name || '', model, time: now.toLocaleString() },
            ai: { results: aiResults, time: now.toLocaleString() }
          };
          return { ...item, rounds: [...item.rounds, round], preview: aiResults.original || item.preview };
        }
        return item;
      });
      session = newHistory.find(item => item.id === activeId);
    } else {
      // 新会话
      const id = now.getTime();
      const round = {
        user: { filename: file?.name || '', model, time: now.toLocaleString() },
        ai: { results: aiResults, time: now.toLocaleString() }
      };
      session = {
        id,
        time: now.toLocaleString(),
        model,
        rounds: [round],
        preview: aiResults.original
      };
      newHistory = [session, ...newHistory].slice(0, 20);
      setActiveId(id);
    }
    setHistory(newHistory);
    localStorage.setItem('analysis_history', JSON.stringify(newHistory));
  };

  // 当前会话
  const session = activeId ? history.find(h => h.id === activeId) : null;

  // 语言切换
  const handleLang = (lng) => i18n.changeLanguage(lng);

  // 模型下拉菜单
  const modelMenu = (
    <Menu onClick={handleModelChange} selectedKeys={[model]} className="model-dropdown">
      {MODELS.map(m => <Menu.Item key={m.key}>{m.label}</Menu.Item>)}
    </Menu>
  );

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={270} className="sidebar" style={{ minHeight: '100vh', background: '#fff', boxShadow: '2px 0 8px #0001', zIndex: 10 }}>
        <Sidebar
          history={history}
          activeId={activeId}
          onSelect={handleSelect}
          onNew={handleNew}
          onClear={handleClear}
        />
      </Sider>
      <Layout>
        <Header className="ant-layout-header" style={{ display: 'flex', alignItems: 'center', height: 64, background: 'var(--btn-gradient)' }}>
          <div className="header-title">Particle Analysis System</div>
          <div className="header-actions">
            <Dropdown overlay={modelMenu} trigger={["click"]}>
              <Button className="model-dropdown" style={{ marginRight: 12 }}>{model}</Button>
            </Dropdown>
            <Button className="theme-toggle" onClick={toggleTheme} icon={theme === 'dark' ? <BulbFilled /> : <BulbOutlined />} />
            <Button className="lang-switch" onClick={() => handleLang('zh')}>中</Button>
            <Button className="lang-switch" onClick={() => handleLang('en')}>EN</Button>
          </div>
        </Header>
        <Content style={{ background: theme === 'dark' ? '#181a1b' : '#f8fafc', minHeight: 'calc(100vh - 64px)' }}>
          <ChatMain session={session} loading={loading} onAnalyze={onAnalyze} model={model} />
        </Content>
      </Layout>
    </Layout>
  );
} 
