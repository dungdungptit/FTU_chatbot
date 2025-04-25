/* eslint-disable no-underscore-dangle */
import { IFqasRecord } from '@/models/fqas';
import { ip } from '@/utils/ip';
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
  Tabs,
  Divider,
} from 'antd';
import TextArea from 'antd/lib/input/TextArea';
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

const FormQuestion = () => {
  const [form] = Form.useForm();
  const [status, setStatus] = useState<boolean>(false);
  const fqasModel = useModel('fqas');
  const pathname = window.location.pathname;
  const IntentId = pathname.split('/')[4];

  const handleFinish = async (values: IFqasRecord) => {
    if (fqasModel.edit) {
      if (fqasModel?.record?.id) {
        const payload = {
          data: {
            fqas: values.question,
            chatbot_answer: values.chatbot_answer,
            human_answer: values.human_answer,
            like: values.like,
            used: values.used
          }
        }
        await fqasModel.upd({ id: fqasModel?.record?.id, data: payload });
        fqasModel.setVisibleForm(false);
        fqasModel.setEdit(false);
        fqasModel.getDataById(fqasModel?.record?.id ?? '');
        fqasModel.getData({ intent_id: IntentId });
      } else {
        const payload = {
          data: {
            fqas: values.question,
            chatbot_answer: values.chatbot_answer,
            human_answer: values.human_answer,
            like: values.like,
            used: values.used
          }
        }
        await fqasModel.add(payload);
        fqasModel.setVisibleForm(false);
        fqasModel.setEdit(false);
        fqasModel.getData({ intent_id: IntentId });
      }
    } else {
      const payload = {
        data: {
          fqas: values.question,
          chatbot_answer: values.chatbot_answer,
          human_answer: values.human_answer,
          like: values.like,
          used: values.used
        }
      }
      await fqasModel.add(payload);
      fqasModel.setVisibleForm(false);
      fqasModel.setEdit(false);
      fqasModel.getData({ intent_id: IntentId });
    }
  };

  const handleFinishFile = async (values: any) => {
    const data = document.getElementById('formQuestionFile') as HTMLFormElement;
    const formData = new FormData(data);
    formData.append('id', IntentId);
    await fqasModel.addFileQuestion(formData);
    fqasModel.setVisibleForm(false);
    fqasModel.setEdit(false);
    fqasModel.getData({ intent_id: IntentId });
  };

  console.log(fqasModel.record);
  return (
    <Spin spinning={fqasModel.loading}>
      <Card title={'Chỉnh sửa'}>
        <Form
          id="formQuestion"
          // {...layout}
          layout='vertical'
          form={form}
          onFinish={handleFinish}
          initialValues={{
            ...(fqasModel?.record ?? {}),
          }}
        >
          <Form.Item label="Mã câu hỏi" name="fqas_id" hidden={true}>
            <Input
              placeholder="Mã câu hỏi"
              name="fqas_id"
              defaultValue={fqasModel?.record?.id ?? ''}
              disabled
            />
          </Form.Item>
          <Form.Item label="Câu hỏi" name="question">
            <Input placeholder="Nhập câu hỏi" name="question" />
          </Form.Item>
          <Form.Item label="Câu trả lời chatbot" name="chatbot_answer">
            <TextArea placeholder="Nội dung thông tin" style={{
              minHeight: '200px'
            }} />
          </Form.Item>
          <Form.Item label="Phản hồi người dùng" name="human_answer">
            <TextArea placeholder="Nội dung thông tin" style={{
              minHeight: '200px'
            }} />
          </Form.Item>
          <Form.Item label="Đánh giá" name="like">
            <Input placeholder="Nhập câu trả lời" name="like" />
          </Form.Item>
          <Form.Item label="Duyệt" name="used" initialValue={fqasModel.record?.used}>
            <Select options={[
              {
                value: true,
                label: 'Duyệt',
              },
              {
                value: false,
                label: 'Không duyệt',
              },
            ]} />
            {/* <Input placeholder="Nhập câu trả lời" name="used" /> */}
          </Form.Item>
          <Form.Item {...tailLayout}>
            <Button type="primary" htmlType="submit">
              {'Cập nhật'}
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </Spin>
  );
};

export default FormQuestion;
