/**
 * Tests for DragDropProvider
 * اختبارات مزود السحب والإفلات
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DragDropProvider, useDragDrop, DropZone } from '../DragDropProvider';

describe('DragDropProvider', () => {
  const TestConsumer = () => {
    const { state, isDragging, canDrop, progress, error } = useDragDrop();
    return (
      <div>
        <span data-testid="state">{state}</span>
        <span data-testid="isDragging">{isDragging.toString()}</span>
        <span data-testid="canDrop">{canDrop.toString()}</span>
        {progress !== null && <span data-testid="progress">{progress}</span>}
        {error && <span data-testid="error">{error}</span>}
      </div>
    );
  };

  const TestComponent = (props: any) => (
    <DragDropProvider {...props}>
      <TestConsumer />
      <div data-testid="dropzone">Drop Zone</div>
    </DragDropProvider>
  );

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should initialize with idle state', () => {
    render(<TestComponent />);

    expect(screen.getByTestId('state').textContent).toBe('idle');
    expect(screen.getByTestId('isDragging').textContent).toBe('false');
    expect(screen.getByTestId('canDrop').textContent).toBe('false');
  });

  it('should detect drag enter', () => {
    render(<TestComponent />);
    const dropzone = screen.getByTestId('dropzone');

    fireEvent.dragEnter(dropzone, {
      dataTransfer: {
        types: ['Files'],
      },
    });

    expect(screen.getByTestId('state').textContent).toBe('dragging');
    expect(screen.getByTestId('isDragging').textContent).toBe('true');
  });

  it('should handle drag leave', () => {
    render(<TestComponent />);
    const dropzone = screen.getByTestId('dropzone');

    fireEvent.dragEnter(dropzone, {
      dataTransfer: {
        types: ['Files'],
      },
    });

    fireEvent.dragLeave(dropzone);

    expect(screen.getByTestId('state').textContent).toBe('idle');
  });

  it('should handle drop', async () => {
    const onDrop = jest.fn();
    const file = new File(['content'], 'test.txt', { type: 'text/plain' });

    render(<TestComponent onDrop={onDrop} />);
    const dropzone = screen.getByTestId('dropzone');

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files: [file],
        types: ['Files'],
      },
    });

    await waitFor(() => {
      expect(onDrop).toHaveBeenCalled();
    });

    const result = onDrop.mock.calls[0][0];
    expect(result.acceptedFiles).toHaveLength(1);
    expect(result.totalFiles).toBe(1);
  });

  it('should validate file types', async () => {
    const onDrop = jest.fn();
    const file = new File(['content'], 'test.pdf', { type: 'application/pdf' });

    render(
      <DragDropProvider
        allowedTypes={{
          extensions: ['txt'],
          mimeTypes: ['text/plain'],
        }}
        onDrop={onDrop}
      >
        <TestConsumer />
        <div data-testid="dropzone">Drop Zone</div>
      </DragDropProvider>
    );

    const dropzone = screen.getByTestId('dropzone');

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files: [file],
        types: ['Files'],
      },
    });

    await waitFor(() => {
      expect(onDrop).toHaveBeenCalled();
    });

    const result = onDrop.mock.calls[0][0];
    expect(result.rejectedFiles).toHaveLength(1);
    expect(result.rejectedFiles[0].reason).toContain('pdf');
  });

  it('should validate file size', async () => {
    const onDrop = jest.fn();
    // Create a 2MB file
    const file = new File([new ArrayBuffer(2 * 1024 * 1024)], 'large.txt', { type: 'text/plain' });

    render(
      <DragDropProvider
        allowedTypes={{
          maxSize: 1024 * 1024, // 1MB max
        }}
        onDrop={onDrop}
      >
        <TestConsumer />
        <div data-testid="dropzone">Drop Zone</div>
      </DragDropProvider>
    );

    const dropzone = screen.getByTestId('dropzone');

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files: [file],
        types: ['Files'],
      },
    });

    await waitFor(() => {
      expect(onDrop).toHaveBeenCalled();
    });

    const result = onDrop.mock.calls[0][0];
    expect(result.rejectedFiles).toHaveLength(1);
    expect(result.rejectedFiles[0].reason).toContain('يتجاوز');
  });

  it('should handle multiple files', async () => {
    const onDrop = jest.fn();
    const files = [
      new File(['content1'], 'file1.txt', { type: 'text/plain' }),
      new File(['content2'], 'file2.txt', { type: 'text/plain' }),
    ];

    render(<TestComponent onDrop={onDrop} multiple={true} />);
    const dropzone = screen.getByTestId('dropzone');

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files,
        types: ['Files'],
      },
    });

    await waitFor(() => {
      expect(onDrop).toHaveBeenCalled();
    });

    const result = onDrop.mock.calls[0][0];
    expect(result.acceptedFiles).toHaveLength(2);
  });

  it('should limit to single file when multiple is false', async () => {
    const onDrop = jest.fn();
    const files = [
      new File(['content1'], 'file1.txt', { type: 'text/plain' }),
      new File(['content2'], 'file2.txt', { type: 'text/plain' }),
    ];

    render(<TestComponent onDrop={onDrop} multiple={false} />);
    const dropzone = screen.getByTestId('dropzone');

    fireEvent.drop(dropzone, {
      dataTransfer: {
        files,
        types: ['Files'],
      },
    });

    await waitFor(() => {
      expect(onDrop).toHaveBeenCalled();
    });

    const result = onDrop.mock.calls[0][0];
    expect(result.acceptedFiles).toHaveLength(1);
  });

  it('should call drag callbacks', () => {
    const onDragEnter = jest.fn();
    const onDragOver = jest.fn();
    const onDragLeave = jest.fn();

    render(
      <DragDropProvider
        onDragEnter={onDragEnter}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
      >
        <TestConsumer />
        <div data-testid="dropzone">Drop Zone</div>
      </DragDropProvider>
    );

    const dropzone = screen.getByTestId('dropzone');

    fireEvent.dragEnter(dropzone, {
      dataTransfer: {
        types: ['Files'],
      },
    });
    expect(onDragEnter).toHaveBeenCalled();

    fireEvent.dragOver(dropzone);
    expect(onDragOver).toHaveBeenCalled();

    fireEvent.dragLeave(dropzone);
    expect(onDragLeave).toHaveBeenCalled();
  });

  it('should throw error if useDragDrop used outside provider', () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    const InvalidComponent = () => {
      useDragDrop();
      return <div />;
    };

    expect(() => {
      render(<InvalidComponent />);
    }).toThrow('useDragDrop must be used within a DragDropProvider');

    consoleSpy.mockRestore();
  });
});

describe('DropZone', () => {
  it('should render with default hint text', () => {
    render(
      <DragDropProvider>
        <DropZone />
      </DragDropProvider>
    );

    expect(screen.getByText('اسحب الملفات هنا أو انقر للاختيار')).toBeInTheDocument();
  });

  it('should render custom hint text', () => {
    render(
      <DragDropProvider>
        <DropZone hintText="Custom hint" />
      </DragDropProvider>
    );

    expect(screen.getByText('Custom hint')).toBeInTheDocument();
  });

  it('should show active text during drag', () => {
    render(
      <DragDropProvider>
        <DropZone />
      </DragDropProvider>
    );

    const dropzone = screen.getByText('اسحب الملفات هنا أو انقر للاختيار');

    fireEvent.dragEnter(dropzone.parentElement!, {
      dataTransfer: {
        types: ['Files'],
      },
    });

    expect(screen.getByText('أفلت الملفات هنا')).toBeInTheDocument();
  });
});
