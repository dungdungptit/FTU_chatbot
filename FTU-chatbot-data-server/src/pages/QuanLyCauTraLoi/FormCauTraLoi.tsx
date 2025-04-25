/* eslint-disable no-underscore-dangle */
import answer, { IAnswerRecord } from '@/models/answer';
import rules from '@/utils/rules';
import {
  Form,
  Input,
  Button,
  Card,
  Spin,
  DatePicker,
  InputNumber,
  Col,
  Row,
  Select,
  Upload,
  Tabs,
} from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import React, { useEffect, useState } from 'react';
import { useModel } from 'umi';

const layout = {
  labelCol: { span: 6 },
  wrapperCol: { span: 16 },
};
const tailLayout = {
  wrapperCol: { offset: 8, span: 16 },
};

const { TabPane } = Tabs;

const normFile = (e: any) => {
  console.log('Upload event:', e);

  if (Array.isArray(e)) {
    return e;
  }
  return e && e.fileList;
};

const FormAnswer = () => {
  const [form] = Form.useForm();
  const [status, setStatus] = useState<boolean>(false);
  const answerModel = useModel('answer');
  const pathname = window.location.pathname;
  const IntentId = pathname.split('/')[2];

  const handleFinish = async (values: any) => {
    if (answerModel.edit) {
      if (answerModel?.record?.id) {
        const data = document.getElementById('formAnswer') as HTMLFormElement;
        const formData = new FormData(data);
        formData.append('id_answer', answerModel?.record?.id_answer ?? '');
        formData.append('intent_id', IntentId);
        if (answerModel.record?.type_answer === 'text') {
          formData.delete('file');
          await answerModel.upd({ id: answerModel?.record?.id_answer, data: formData });
        } else if (answerModel.record?.type_answer === 'image') {
          console.log('Call API update image');
          await answerModel.updWithImage({ id: answerModel?.record?.id_answer, data: formData });
        }
        // await answerModel.upd({
        //   ...answerModel?.record,
        //   ...values,
        //   id: answerModel?.record?.id,
        // });
        answerModel.setVisibleForm(false);
        answerModel.setEdit(false);
        // answerModel.getDataById(answerModel?.record?.id ?? '');
        answerModel.getData({ intent_id: IntentId });
      } else {
        const data = document.getElementById('formAnswer') as HTMLFormElement;
        const formData = new FormData(data);
        formData.append('intent_id', IntentId);
        if (values?.type === 'image') {
          await answerModel.addWithImage(formData);
        } else {
          formData.delete('file');
          console.log('Call API add text');
          await answerModel.add(formData);
        }

        answerModel.setVisibleForm(false);
        answerModel.setEdit(false);
        answerModel.getData({ intent_id: IntentId });
      }
    } else {
      const data = document.getElementById('formAnswer') as HTMLFormElement;
      const formData = new FormData(data);
      formData.append('intent_id', IntentId);
      console.log('values', values);

      if (values?.type === 'image') {
        await answerModel.addWithImage(formData);
      } else {
        formData.delete('file');
        console.log('Call API add text');
        await answerModel.add(formData);
      }

      answerModel.setVisibleForm(false);
      answerModel.setEdit(false);
      answerModel.getData({ intent_id: IntentId });
    }
  };

  const handleFinishFile = async (values: any) => {};

  return (
    <Spin spinning={answerModel.loading}>
      <Card title={answerModel.edit ? 'Chỉnh sửa' : 'Thêm mới'}>
        <Form
          id="formAnswer"
          {...layout}
          form={form}
          onFinish={handleFinish}
          initialValues={{
            ...(answerModel?.record ?? {}),
          }}
        >
          <Form.Item label="Mã câu hỏi" name="answer_id" hidden={true}>
            <Input
              placeholder="Mã câu hỏi"
              name="answer_id"
              defaultValue={answerModel?.record?.id ?? ''}
              disabled
            />
          </Form.Item>
          <Form.Item label="Thứ tự" name="sort">
            <Input
              placeholder="Thứ tự"
              name="sort"
              type="number"
              min={1}
              defaultValue={answerModel?.record?.sort ?? 1}
            />
          </Form.Item>
          <Form.Item label="Chọn loại câu trả lời" name="type">
            <Select
              placeholder="Chọn loại câu trả lời"
              name="type"
              defaultValue={answerModel?.record?.type_answer ?? 'text'}
              onChange={(value) => {
                if (value === 'image') {
                  answerModel.setRecord({ ...answerModel?.record, type_answer: 'image' });
                } else {
                  answerModel.setRecord({ ...answerModel?.record, type_answer: 'text' });
                }
              }}
            >
              <Select.Option value="text"> Văn bản </Select.Option>
              <Select.Option value="image"> Hình ảnh </Select.Option>
            </Select>
          </Form.Item>

          {answerModel?.record?.type_answer === 'image' ? (
            <Form.Item label="Hình ảnh" name="file">
              <Input type="file" name="file" />
            </Form.Item>
          ) : (
            <Form.Item label="Câu trả lời" name="answer">
              <Input.TextArea
                showCount
                maxLength={500}
                placeholder="Nhập câu trả lời"
                name="answer"
              />
            </Form.Item>
          )}

          <Form.Item {...tailLayout}>
            <Button type="primary" htmlType="submit">
              {answerModel.edit ? 'Cập nhật' : 'Thêm mới'}
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </Spin>
  );
};

export default FormAnswer;
