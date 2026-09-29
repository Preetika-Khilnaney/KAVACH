import { createContext, useContext, useState } from 'react';
import type { ReactNode } from 'react';
import type { PipelineEvent } from '../api/websocket';
import { usePipelineStream } from '../api/websocket';

interface IngestionSession {
  selectedFile: File | null;
  isProcessing: boolean;
  progressPercent: number;
  activeStageIdx: number;
  isComplete: boolean;
  statusText: string;
  capturePath: string | null;
  setSelectedFile: (file: File | null) => void;
  setIsProcessing: (value: boolean) => void;
  setProgressPercent: (value: number) => void;
  setActiveStageIdx: (value: number) => void;
  setIsComplete: (value: boolean) => void;
  setStatusText: (value: string) => void;
  setCapturePath: (value: string | null) => void;
  graphEvents: PipelineEvent[];
  graphConnected: boolean;
  graphReplaying: boolean;
}

const IngestionContext = createContext<IngestionSession | undefined>(undefined);

export function IngestionProvider({ children }: { children: ReactNode }) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [activeStageIdx, setActiveStageIdx] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [statusText, setStatusText] = useState('');
  const [capturePath, setCapturePath] = useState<string | null>(null);
  const { events: graphEvents, connected: graphConnected, replaying: graphReplaying } = usePipelineStream(capturePath);

  return (
    <IngestionContext.Provider value={{
      selectedFile,
      isProcessing,
      progressPercent,
      activeStageIdx,
      isComplete,
      statusText,
      capturePath,
      setSelectedFile,
      setIsProcessing,
      setProgressPercent,
      setActiveStageIdx,
      setIsComplete,
      setStatusText,
      setCapturePath,
      graphEvents,
      graphConnected,
      graphReplaying,
    }}>
      {children}
    </IngestionContext.Provider>
  );
}

export function useIngestion() {
  const context = useContext(IngestionContext);
  if (!context) {
    throw new Error('useIngestion must be used within IngestionProvider');
  }
  return context;
}
