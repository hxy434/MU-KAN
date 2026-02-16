import React from 'react';
import { Button } from 'antd';
import { useTranslation } from 'react-i18next';

export default function LanguageSwitcher() {
  const { i18n } = useTranslation();
  return (
    <div>
      <Button onClick={() => i18n.changeLanguage('zh')}>中文</Button>
      <Button onClick={() => i18n.changeLanguage('en')} style={{ marginLeft: 8 }}>English</Button>
    </div>
  );
} 